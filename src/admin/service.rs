//! Admin API 业务逻辑服务

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::kiro::model::credentials::KiroCredentials;
use crate::kiro::token_manager::MultiTokenManager;

use super::error::AdminServiceError;
use super::types::{
    AddCredentialRequest, AddCredentialResponse, AdminHealthResponse, BalanceResponse,
    CredentialStatusItem, CredentialsStatusResponse, LoadBalancingModeResponse,
    SetLoadBalancingModeRequest,
};

/// 余额缓存过期时间（秒），5 分钟
const BALANCE_CACHE_TTL_SECS: i64 = 300;

/// 缓存的余额条目（含时间戳）
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedBalance {
    /// 缓存时间（Unix 秒）
    cached_at: f64,
    /// 缓存的余额数据
    data: BalanceResponse,
}

/// Admin 服务
///
/// 封装所有 Admin API 的业务逻辑
pub struct AdminService {
    token_manager: Arc<MultiTokenManager>,
    balance_cache: Mutex<HashMap<u64, CachedBalance>>,
    cache_path: Option<PathBuf>,
    /// 已注册的端点名称集合（用于 add_credential 校验）
    known_endpoints: HashSet<String>,
}

impl AdminService {
    pub fn new(
        token_manager: Arc<MultiTokenManager>,
        known_endpoints: impl IntoIterator<Item = String>,
    ) -> Self {
        let cache_path = token_manager
            .cache_dir()
            .map(|d| d.join("kiro_balance_cache.json"));

        let balance_cache = Self::load_balance_cache_from(&cache_path);

        Self {
            token_manager,
            balance_cache: Mutex::new(balance_cache),
            cache_path,
            known_endpoints: known_endpoints.into_iter().collect(),
        }
    }

    /// 获取所有凭据状态
    pub fn get_all_credentials(&self) -> CredentialsStatusResponse {
        let snapshot = self.token_manager.snapshot();
        let default_endpoint = self.token_manager.config().default_endpoint.clone();
        let status_summary = CredentialStatusSummary::from_entries(&snapshot.entries);

        let mut credentials: Vec<CredentialStatusItem> = snapshot
            .entries
            .into_iter()
            .map(|entry| CredentialStatusItem {
                id: entry.id,
                priority: entry.priority,
                disabled: entry.disabled,
                failure_count: entry.failure_count,
                is_current: entry.id == snapshot.current_id,
                expires_at: entry.expires_at,
                auth_method: entry.auth_method,
                has_profile_arn: entry.has_profile_arn,
                refresh_token_hash: entry.refresh_token_hash,
                api_key_hash: entry.api_key_hash,
                masked_api_key: entry.masked_api_key,
                email: entry.email,
                success_count: entry.success_count,
                metered_credits: entry.metered_credits,
                metered_request_count: entry.metered_request_count,
                metering_started_at: entry.metering_started_at,
                last_used_at: entry.last_used_at.clone(),
                has_proxy: entry.has_proxy,
                proxy_url: entry.proxy_url,
                refresh_failure_count: entry.refresh_failure_count,
                quota_exhaustion_reason: quota_exhaustion_reason(&entry.disabled_reason),
                disabled_reason: entry.disabled_reason,
                disabled_until: entry.disabled_until,
                suspend_streak: entry.suspend_streak,
                probation: entry.probation,
                endpoint: entry.endpoint.unwrap_or_else(|| default_endpoint.clone()),
            })
            .collect();

        // 按优先级排序（数字越小优先级越高）
        credentials.sort_by_key(|c| c.priority);

        CredentialsStatusResponse {
            total: snapshot.total,
            available: snapshot.available,
            disabled_count: status_summary.disabled_count,
            current_id: snapshot.current_id,
            all_disabled: status_summary.all_disabled,
            unavailable_reason: status_summary.unavailable_reason,
            disabled_reason_counts: status_summary.disabled_reason_counts,
            monthly_request_count_disabled: status_summary.monthly_request_count_disabled,
            credentials,
        }
    }

    /// 获取 Admin 健康状态
    pub fn get_health(&self) -> AdminHealthResponse {
        let snapshot = self.token_manager.snapshot();
        let status_summary = CredentialStatusSummary::from_entries(&snapshot.entries);

        AdminHealthResponse {
            status: status_summary.status,
            total: snapshot.total,
            available: snapshot.available,
            disabled_count: status_summary.disabled_count,
            current_id: snapshot.current_id,
            all_disabled: status_summary.all_disabled,
            unavailable_reason: status_summary.unavailable_reason,
            disabled_reason_counts: status_summary.disabled_reason_counts,
            monthly_request_count_disabled: status_summary.monthly_request_count_disabled,
        }
    }

    /// 设置凭据禁用状态
    pub fn set_disabled(&self, id: u64, disabled: bool) -> Result<(), AdminServiceError> {
        // 先获取当前凭据 ID，用于判断是否需要切换
        let snapshot = self.token_manager.snapshot();
        let current_id = snapshot.current_id;

        self.token_manager
            .set_disabled(id, disabled)
            .map_err(|e| self.classify_error(e, id))?;

        // 只有禁用的是当前凭据时才尝试切换到下一个
        if disabled && id == current_id {
            let _ = self.token_manager.switch_to_next();
        }
        Ok(())
    }

    /// 设置凭据优先级
    pub fn set_priority(&self, id: u64, priority: u32) -> Result<(), AdminServiceError> {
        self.token_manager
            .set_priority(id, priority)
            .map_err(|e| self.classify_error(e, id))
    }

    /// 重置失败计数并重新启用
    pub fn reset_and_enable(&self, id: u64) -> Result<(), AdminServiceError> {
        self.token_manager
            .reset_and_enable(id)
            .map_err(|e| self.classify_error(e, id))
    }

    /// 获取凭据余额（带缓存）
    pub async fn get_balance(
        &self,
        id: u64,
        force_refresh: bool,
    ) -> Result<BalanceResponse, AdminServiceError> {
        // 先查缓存
        if !force_refresh {
            let cache = self.balance_cache.lock();
            if let Some(cached) = cache.get(&id) {
                let now = Utc::now().timestamp() as f64;
                if (now - cached.cached_at) < BALANCE_CACHE_TTL_SECS as f64 {
                    tracing::debug!("凭据 #{} 余额命中缓存", id);
                    return Ok(cached.data.clone());
                }
            }
        }

        // 缓存未命中或已过期，从上游获取
        let balance = self.fetch_balance(id).await?;

        // 更新缓存
        {
            let mut cache = self.balance_cache.lock();
            cache.insert(
                id,
                CachedBalance {
                    cached_at: Utc::now().timestamp() as f64,
                    data: balance.clone(),
                },
            );
        }
        self.save_balance_cache();

        Ok(balance)
    }

    /// 从上游获取余额（无缓存）
    async fn fetch_balance(&self, id: u64) -> Result<BalanceResponse, AdminServiceError> {
        let usage = self
            .token_manager
            .get_usage_limits_for(id)
            .await
            .map_err(|e| self.classify_balance_error(e, id))?;

        let current_usage = usage.current_usage();
        let usage_limit = usage.usage_limit();
        let remaining = (usage_limit - current_usage).max(0.0);
        let usage_percentage = if usage_limit > 0.0 {
            (current_usage / usage_limit * 100.0).min(100.0)
        } else {
            0.0
        };
        let reconciliation = self
            .token_manager
            .reconcile_usage(id, current_usage, usage.next_date_reset)
            .ok_or(AdminServiceError::NotFound { id })?;

        Ok(BalanceResponse {
            id,
            subscription_title: usage.subscription_title().map(|s| s.to_string()),
            current_usage,
            usage_limit,
            remaining,
            usage_percentage,
            next_reset_at: usage.next_date_reset,
            local_credits_delta: reconciliation.local_credits_delta,
            local_metered_request_count: reconciliation.local_request_count_delta,
            source_usage_delta: reconciliation.source_usage_delta,
            unattributed_usage_delta: reconciliation.unattributed_delta,
            reconciliation_baseline_at: reconciliation.baseline_at,
        })
    }

    /// 添加新凭据
    pub async fn add_credential(
        &self,
        req: AddCredentialRequest,
    ) -> Result<AddCredentialResponse, AdminServiceError> {
        // 校验端点名：未指定则默认合法，指定则必须已注册
        if let Some(ref name) = req.endpoint {
            if !self.known_endpoints.contains(name) {
                let mut known: Vec<&str> =
                    self.known_endpoints.iter().map(|s| s.as_str()).collect();
                known.sort();
                return Err(AdminServiceError::InvalidCredential(format!(
                    "未知端点 \"{}\"，已注册端点: {:?}",
                    name, known
                )));
            }
        }

        // 构建凭据对象
        let email = req.email.clone();
        let new_cred = KiroCredentials {
            id: None,
            access_token: None,
            refresh_token: req.refresh_token,
            profile_arn: None,
            expires_at: None,
            auth_method: Some(req.auth_method),
            client_id: req.client_id,
            client_secret: req.client_secret,
            priority: req.priority,
            region: req.region,
            auth_region: req.auth_region,
            api_region: req.api_region,
            machine_id: req.machine_id,
            email: req.email,
            subscription_title: None, // 将在首次获取使用额度时自动更新
            proxy_url: req.proxy_url,
            proxy_username: req.proxy_username,
            proxy_password: req.proxy_password,
            disabled: false, // 新添加的凭据默认启用
            kiro_api_key: req.kiro_api_key,
            endpoint: req.endpoint,
        };

        // 调用 token_manager 添加凭据
        let credential_id = self
            .token_manager
            .add_credential(new_cred)
            .await
            .map_err(|e| self.classify_add_error(e))?;

        // 主动获取订阅等级，避免首次请求时 Free 账号绕过 Opus 模型过滤
        if let Err(e) = self.token_manager.get_usage_limits_for(credential_id).await {
            tracing::warn!("添加凭据后获取订阅等级失败（不影响凭据添加）: {}", e);
        }

        Ok(AddCredentialResponse {
            success: true,
            message: format!("凭据添加成功，ID: {}", credential_id),
            credential_id,
            email,
        })
    }

    /// 删除凭据
    pub fn delete_credential(&self, id: u64) -> Result<(), AdminServiceError> {
        self.token_manager
            .delete_credential(id)
            .map_err(|e| self.classify_delete_error(e, id))?;

        // 清理已删除凭据的余额缓存
        {
            let mut cache = self.balance_cache.lock();
            cache.remove(&id);
        }
        self.save_balance_cache();

        Ok(())
    }

    /// 获取负载均衡模式
    pub fn get_load_balancing_mode(&self) -> LoadBalancingModeResponse {
        LoadBalancingModeResponse {
            mode: self.token_manager.get_load_balancing_mode(),
        }
    }

    /// 设置负载均衡模式
    pub fn set_load_balancing_mode(
        &self,
        req: SetLoadBalancingModeRequest,
    ) -> Result<LoadBalancingModeResponse, AdminServiceError> {
        // 验证模式值
        if req.mode != "priority" && req.mode != "balanced" {
            return Err(AdminServiceError::InvalidCredential(
                "mode 必须是 'priority' 或 'balanced'".to_string(),
            ));
        }

        self.token_manager
            .set_load_balancing_mode(req.mode.clone())
            .map_err(|e| AdminServiceError::InternalError(e.to_string()))?;

        Ok(LoadBalancingModeResponse { mode: req.mode })
    }

    /// 强制刷新指定凭据的 Token
    pub async fn force_refresh_token(&self, id: u64) -> Result<(), AdminServiceError> {
        self.token_manager
            .force_refresh_token_for(id)
            .await
            .map_err(|e| self.classify_balance_error(e, id))
    }

    // ============ 余额缓存持久化 ============

    fn load_balance_cache_from(cache_path: &Option<PathBuf>) -> HashMap<u64, CachedBalance> {
        let path = match cache_path {
            Some(p) => p,
            None => return HashMap::new(),
        };

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return HashMap::new(),
        };

        // 文件中使用字符串 key 以兼容 JSON 格式
        let map: HashMap<String, CachedBalance> = match serde_json::from_str(&content) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("解析余额缓存失败，将忽略: {}", e);
                return HashMap::new();
            }
        };

        let now = Utc::now().timestamp() as f64;
        map.into_iter()
            .filter_map(|(k, v)| {
                let id = k.parse::<u64>().ok()?;
                // 丢弃超过 TTL 的条目
                if (now - v.cached_at) < BALANCE_CACHE_TTL_SECS as f64 {
                    Some((id, v))
                } else {
                    None
                }
            })
            .collect()
    }

    fn save_balance_cache(&self) {
        let path = match &self.cache_path {
            Some(p) => p,
            None => return,
        };

        // 持有锁期间完成序列化和写入，防止并发损坏
        let cache = self.balance_cache.lock();
        let map: HashMap<String, &CachedBalance> =
            cache.iter().map(|(k, v)| (k.to_string(), v)).collect();

        match serde_json::to_string_pretty(&map) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    tracing::warn!("保存余额缓存失败: {}", e);
                }
            }
            Err(e) => tracing::warn!("序列化余额缓存失败: {}", e),
        }
    }

    // ============ 错误分类 ============

    /// 分类简单操作错误（set_disabled, set_priority, reset_and_enable）
    fn classify_error(&self, e: anyhow::Error, id: u64) -> AdminServiceError {
        let msg = e.to_string();
        if msg.contains("不存在") {
            AdminServiceError::NotFound { id }
        } else {
            AdminServiceError::InternalError(msg)
        }
    }

    /// 分类余额查询错误（可能涉及上游 API 调用）
    fn classify_balance_error(&self, e: anyhow::Error, id: u64) -> AdminServiceError {
        let msg = e.to_string();

        // 1. 凭据不存在
        if msg.contains("不存在") {
            return AdminServiceError::NotFound { id };
        }

        // 2. API Key 凭据不支持刷新：客户端请求错误，映射为 400
        if msg.contains("API Key 凭据不支持刷新") {
            return AdminServiceError::InvalidCredential(msg);
        }

        // 3. 上游服务错误特征：HTTP 响应错误或网络错误
        let is_upstream_error =
            // HTTP 响应错误（来自 refresh_*_token 的错误消息）
            msg.contains("凭证已过期或无效") ||
            msg.contains("权限不足") ||
            msg.contains("已被限流") ||
            msg.contains("服务器错误") ||
            msg.contains("Token 刷新失败") ||
            msg.contains("暂时不可用") ||
            // 网络错误（reqwest 错误）
            msg.contains("error trying to connect") ||
            msg.contains("connection") ||
            msg.contains("timeout") ||
            msg.contains("timed out");

        if is_upstream_error {
            AdminServiceError::UpstreamError(msg)
        } else {
            // 4. 默认归类为内部错误（本地验证失败、配置错误等）
            // 包括：缺少 refreshToken、refreshToken 已被截断、无法生成 machineId 等
            AdminServiceError::InternalError(msg)
        }
    }

    /// 分类添加凭据错误
    fn classify_add_error(&self, e: anyhow::Error) -> AdminServiceError {
        let msg = e.to_string();

        // 凭据验证失败（refreshToken 无效、格式错误等）
        let is_invalid_credential = msg.contains("缺少 refreshToken")
            || msg.contains("refreshToken 为空")
            || msg.contains("refreshToken 已被截断")
            || msg.contains("凭据已存在")
            || msg.contains("refreshToken 重复")
            || msg.contains("kiroApiKey 重复")
            || msg.contains("缺少 kiroApiKey")
            || msg.contains("kiroApiKey 为空")
            || msg.contains("凭证已过期或无效")
            || msg.contains("权限不足")
            || msg.contains("已被限流");

        if is_invalid_credential {
            AdminServiceError::InvalidCredential(msg)
        } else if msg.contains("error trying to connect")
            || msg.contains("connection")
            || msg.contains("timeout")
        {
            AdminServiceError::UpstreamError(msg)
        } else {
            AdminServiceError::InternalError(msg)
        }
    }

    /// 分类删除凭据错误
    fn classify_delete_error(&self, e: anyhow::Error, id: u64) -> AdminServiceError {
        let msg = e.to_string();
        if msg.contains("不存在") {
            AdminServiceError::NotFound { id }
        } else if msg.contains("只能删除已禁用的凭据") || msg.contains("请先禁用凭据")
        {
            AdminServiceError::InvalidCredential(msg)
        } else {
            AdminServiceError::InternalError(msg)
        }
    }
}

struct CredentialStatusSummary {
    status: String,
    disabled_count: usize,
    all_disabled: bool,
    unavailable_reason: Option<String>,
    disabled_reason_counts: BTreeMap<String, usize>,
    monthly_request_count_disabled: usize,
}

impl CredentialStatusSummary {
    fn from_entries(entries: &[crate::kiro::token_manager::CredentialEntrySnapshot]) -> Self {
        let total = entries.len();
        let disabled_count = entries.iter().filter(|entry| entry.disabled).count();
        let available = total.saturating_sub(disabled_count);
        let all_disabled = total > 0 && available == 0;
        let mut disabled_reason_counts = BTreeMap::new();

        for reason in entries
            .iter()
            .filter(|entry| entry.disabled)
            .filter_map(|entry| entry.disabled_reason.as_deref())
        {
            *disabled_reason_counts
                .entry(reason.to_string())
                .or_insert(0) += 1;
        }

        let monthly_request_count_disabled =
            *disabled_reason_counts.get("QuotaExceeded").unwrap_or(&0);
        let status = if total == 0 || all_disabled {
            "unavailable"
        } else if disabled_count > 0 {
            "degraded"
        } else {
            "ok"
        }
        .to_string();

        let unavailable_reason = if total == 0 {
            Some("NoCredentials".to_string())
        } else if all_disabled && monthly_request_count_disabled == total {
            Some("MONTHLY_REQUEST_COUNT".to_string())
        } else if all_disabled {
            Some("AllCredentialsDisabled".to_string())
        } else {
            None
        };

        Self {
            status,
            disabled_count,
            all_disabled,
            unavailable_reason,
            disabled_reason_counts,
            monthly_request_count_disabled,
        }
    }
}

fn quota_exhaustion_reason(disabled_reason: &Option<String>) -> Option<String> {
    match disabled_reason.as_deref() {
        Some("QuotaExceeded") => Some("MONTHLY_REQUEST_COUNT".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::token_manager::CredentialEntrySnapshot;

    fn snapshot(id: u64, disabled: bool, reason: Option<&str>) -> CredentialEntrySnapshot {
        CredentialEntrySnapshot {
            id,
            priority: id as u32,
            disabled,
            failure_count: 0,
            auth_method: None,
            has_profile_arn: false,
            expires_at: None,
            refresh_token_hash: None,
            api_key_hash: None,
            masked_api_key: None,
            email: None,
            success_count: 0,
            last_used_at: None,
            metered_credits: 0.0,
            metered_request_count: 0,
            metering_started_at: None,
            has_proxy: false,
            proxy_url: None,
            refresh_failure_count: 0,
            disabled_reason: reason.map(str::to_string),
            disabled_until: None,
            suspend_streak: 0,
            probation: false,
            endpoint: None,
        }
    }

    #[test]
    fn status_summary_reports_monthly_request_count_when_all_quota_exhausted() {
        let entries = vec![
            snapshot(1013, true, Some("QuotaExceeded")),
            snapshot(1020, true, Some("QuotaExceeded")),
        ];

        let summary = CredentialStatusSummary::from_entries(&entries);

        assert_eq!(summary.status, "unavailable");
        assert!(summary.all_disabled);
        assert_eq!(
            summary.unavailable_reason.as_deref(),
            Some("MONTHLY_REQUEST_COUNT")
        );
        assert_eq!(summary.monthly_request_count_disabled, 2);
        assert_eq!(
            summary.disabled_reason_counts.get("QuotaExceeded").copied(),
            Some(2)
        );
    }

    #[test]
    fn status_summary_is_degraded_when_some_credentials_remain_available() {
        let entries = vec![
            snapshot(1013, false, None),
            snapshot(1020, true, Some("QuotaExceeded")),
        ];

        let summary = CredentialStatusSummary::from_entries(&entries);

        assert_eq!(summary.status, "degraded");
        assert!(!summary.all_disabled);
        assert_eq!(summary.unavailable_reason, None);
        assert_eq!(summary.disabled_count, 1);
        assert_eq!(summary.monthly_request_count_disabled, 1);
    }

    #[test]
    fn quota_exhaustion_reason_maps_quota_exceeded_to_kiro_reason() {
        assert_eq!(
            quota_exhaustion_reason(&Some("QuotaExceeded".to_string())).as_deref(),
            Some("MONTHLY_REQUEST_COUNT")
        );
        assert_eq!(quota_exhaustion_reason(&Some("Manual".to_string())), None);
    }
}
