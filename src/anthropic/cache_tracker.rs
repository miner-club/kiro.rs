//! 提示词缓存追踪器
//!
//! Kiro 上游按 credits 计费，不返回 Anthropic 语义的 prompt cache 使用量。
//! 本模块在代理侧模拟 Anthropic 的缓存语义：对请求中 `cache_control` 断点
//! 之前的前缀做增量 SHA-256 指纹，按凭据维护带 TTL 的指纹表，命中的部分
//! 报告为 `cache_read_input_tokens`，未命中的部分报告为
//! `cache_creation_input_tokens`，让下游（如 sub2api）的计费贴近直连
//! Anthropic API 的真实成本结构。
//!
//! 移植自 Kiro-Go 的 `proxy/cache_tracker.go`，语义保持一致：
//! - 缓存前缀需达到模型最小可缓存 token 数（默认 1024，Opus 4096）
//! - 命中部分上限为总输入的 85%，保证最新内容不会被报告为全缓存
//! - 出现过显式 `cache_control` 断点后，后续消息尾部成为隐式断点，
//!   使多轮对话能命中此前存储的前缀
//! - Claude Code 每请求漂移的 `x-anthropic-billing-header` 系统块
//!   不参与指纹，避免它破坏缓存命中

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::converter::get_context_window_size;
use super::types::MessagesRequest;

/// 默认缓存 TTL（Anthropic ephemeral 缓存为 5 分钟）
const DEFAULT_PROMPT_CACHE_TTL: Duration = Duration::from_secs(5 * 60);

/// 1 小时 TTL
const ONE_HOUR: Duration = Duration::from_secs(60 * 60);

/// Anthropic 要求缓存前缀达到最小 token 数才会生效。
/// 低于阈值的断点不参与匹配和存储，避免短请求报出不真实的 100% 命中。
const DEFAULT_MIN_CACHEABLE_TOKENS: i32 = 1024;
const OPUS_MIN_CACHEABLE_TOKENS: i32 = 4096;

/// 命中部分占总输入的上限比例。
/// 当前轮的最新内容永远不会全部来自缓存。
const MAX_CACHEABLE_RATIO: f64 = 0.85;

/// 一次请求的模拟缓存使用量
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptCacheUsage {
    pub cache_creation_input_tokens: i32,
    pub cache_read_input_tokens: i32,
    pub cache_creation_5m_input_tokens: i32,
    pub cache_creation_1h_input_tokens: i32,
}

/// 缓存断点：前缀指纹 + 累计 token 数 + TTL
struct PromptCacheBreakpoint {
    fingerprint: [u8; 32],
    cumulative_tokens: i32,
    ttl: Duration,
}

/// 一次请求的缓存画像（由请求内容构建，与凭据无关）
pub struct PromptCacheProfile {
    breakpoints: Vec<PromptCacheBreakpoint>,
    total_input_tokens: i32,
    model: String,
}

impl PromptCacheProfile {
    /// 用外部估算的输入 token 数抬高总量（不低于断点累计值），但不超过模型上下文窗口。
    pub fn raise_total_input_tokens(&mut self, estimated: i32) {
        let capped = estimated.max(0).min(context_window_for_model(&self.model));
        if capped > self.total_input_tokens {
            self.total_input_tokens = capped;
        }
        self.cap_to_context_window();
    }

    fn cap_to_context_window(&mut self) {
        let cap = context_window_for_model(&self.model);
        self.total_input_tokens = self.total_input_tokens.min(cap);
        for breakpoint in &mut self.breakpoints {
            breakpoint.cumulative_tokens = breakpoint.cumulative_tokens.min(cap);
        }
    }
}

/// 请求处理链路中携带的缓存上下文
#[derive(Clone)]
pub struct CacheContext {
    pub tracker: Arc<PromptCacheTracker>,
    pub profile: Arc<PromptCacheProfile>,
}

struct PromptCacheEntry {
    expires_at: Instant,
    ttl: Duration,
}

/// 按凭据维护指纹表的缓存追踪器
#[derive(Default)]
pub struct PromptCacheTracker {
    entries_by_account: Mutex<HashMap<u64, HashMap<[u8; 32], PromptCacheEntry>>>,
}

fn min_cacheable_tokens_for_model(model: &str) -> i32 {
    if model.to_lowercase().contains("opus") {
        OPUS_MIN_CACHEABLE_TOKENS
    } else {
        DEFAULT_MIN_CACHEABLE_TOKENS
    }
}

fn context_window_for_model(model: &str) -> i32 {
    get_context_window_size(model).max(DEFAULT_MIN_CACHEABLE_TOKENS)
}

fn max_reportable_cache_tokens(total_input_tokens: i32) -> i32 {
    ((total_input_tokens.max(0) as f64) * MAX_CACHEABLE_RATIO) as i32
}

impl PromptCacheTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从 Anthropic 请求构建缓存画像。
    ///
    /// 请求中没有任何 `cache_control` 断点时返回 `None`（客户端未启用缓存）。
    pub fn build_claude_profile(&self, req: &MessagesRequest) -> Option<PromptCacheProfile> {
        let blocks = flatten_claude_cache_blocks(req);
        if blocks.is_empty() {
            return None;
        }

        let mut hasher = Sha256::new();
        let mut breakpoints = Vec::new();
        let mut cumulative_tokens = 0i32;
        let mut active_ttl = Duration::ZERO;

        for block in &blocks {
            write_hash_chunk(&mut hasher, &block.canonical);
            cumulative_tokens += block.tokens;

            // 断点判定：
            //   1) 块自身带显式 cache_control；
            //   2) 出现过显式断点后，每个消息尾部成为隐式断点，
            //      让多轮对话能命中此前存储的前缀。
            let breakpoint_ttl = if block.ttl > Duration::ZERO {
                active_ttl = block.ttl;
                block.ttl
            } else if block.is_message_end && active_ttl > Duration::ZERO {
                active_ttl
            } else {
                Duration::ZERO
            };

            if breakpoint_ttl.is_zero() {
                continue;
            }

            let mut fingerprint = [0u8; 32];
            fingerprint.copy_from_slice(&hasher.clone().finalize());
            breakpoints.push(PromptCacheBreakpoint {
                fingerprint,
                cumulative_tokens,
                ttl: breakpoint_ttl,
            });
        }

        if breakpoints.is_empty() {
            return None;
        }

        let mut profile = PromptCacheProfile {
            breakpoints,
            total_input_tokens: cumulative_tokens,
            model: req.model.clone(),
        };
        profile.cap_to_context_window();
        Some(profile)
    }

    /// 计算本次请求的模拟缓存使用量（在请求发出后、依据所选凭据调用）
    ///
    /// 阈值口径：低于模型最小可缓存 token 数的断点视为不存在——不参与命中
    /// 匹配、不计入 cache_creation、也不计入 5m/1h 明细。保证不变式
    /// `cache_creation_5m + cache_creation_1h == cache_creation_input_tokens`，
    /// 下游（sub2api）按明细计费时口径与聚合字段一致。
    pub fn compute(&self, account_id: u64, profile: &PromptCacheProfile) -> PromptCacheUsage {
        if profile.breakpoints.is_empty() {
            return PromptCacheUsage::default();
        }

        let min_tokens = min_cacheable_tokens_for_model(&profile.model);
        // 创建上限取最后一个合格断点（低于阈值的断点不产生缓存创建）
        let Some(last_qualifying) = profile
            .breakpoints
            .iter()
            .rev()
            .find(|b| b.cumulative_tokens >= min_tokens)
        else {
            return PromptCacheUsage::default();
        };
        let mut last_tokens = last_qualifying
            .cumulative_tokens
            .min(profile.total_input_tokens);
        let max_cacheable = max_reportable_cache_tokens(profile.total_input_tokens);
        if last_tokens > max_cacheable {
            last_tokens = max_cacheable;
        }
        let now = Instant::now();

        let mut accounts = self.entries_by_account.lock();
        prune_expired(&mut accounts, now);

        let entries = match accounts.get_mut(&account_id) {
            Some(e) if !e.is_empty() => e,
            _ => {
                // 该凭据首次请求：只报告缓存创建
                let (cache_5m, cache_1h) =
                    compute_ttl_breakdown(profile, 0, last_tokens, min_tokens);
                return PromptCacheUsage {
                    cache_creation_input_tokens: last_tokens,
                    cache_read_input_tokens: 0,
                    cache_creation_5m_input_tokens: cache_5m,
                    cache_creation_1h_input_tokens: cache_1h,
                };
            }
        };

        let mut matched_tokens = 0i32;
        for breakpoint in profile.breakpoints.iter().rev() {
            // 跳过低于最小可缓存阈值的断点
            if breakpoint.cumulative_tokens < min_tokens {
                continue;
            }
            let entry = match entries.get_mut(&breakpoint.fingerprint) {
                Some(e) if e.expires_at > now => e,
                _ => continue,
            };
            // 命中即刷新 TTL（与 Anthropic 语义一致）
            entry.expires_at = now + entry.ttl;
            matched_tokens = breakpoint
                .cumulative_tokens
                .min(profile.total_input_tokens)
                .min(last_tokens);
            break;
        }

        let creation = (last_tokens - matched_tokens).max(0);
        let (cache_5m, cache_1h) =
            compute_ttl_breakdown(profile, matched_tokens, last_tokens, min_tokens);
        debug_assert_eq!(cache_5m + cache_1h, creation);
        PromptCacheUsage {
            cache_creation_input_tokens: creation,
            cache_read_input_tokens: matched_tokens,
            cache_creation_5m_input_tokens: cache_5m,
            cache_creation_1h_input_tokens: cache_1h,
        }
    }

    /// 请求成功后登记本次请求的断点指纹
    pub fn update(&self, account_id: u64, profile: &PromptCacheProfile) {
        if profile.breakpoints.is_empty() {
            return;
        }

        let min_tokens = min_cacheable_tokens_for_model(&profile.model);
        let now = Instant::now();
        let mut accounts = self.entries_by_account.lock();
        prune_expired(&mut accounts, now);

        let entries = accounts.entry(account_id).or_default();
        for breakpoint in &profile.breakpoints {
            if breakpoint.cumulative_tokens < min_tokens {
                continue;
            }
            entries.insert(
                breakpoint.fingerprint,
                PromptCacheEntry {
                    expires_at: now + breakpoint.ttl,
                    ttl: breakpoint.ttl,
                },
            );
        }
    }
}

fn prune_expired(accounts: &mut HashMap<u64, HashMap<[u8; 32], PromptCacheEntry>>, now: Instant) {
    accounts.retain(|_, entries| {
        entries.retain(|_, entry| entry.expires_at > now);
        !entries.is_empty()
    });
}

/// 按 TTL 归类未命中部分的缓存创建量：(5m, 1h)
///
/// 只统计合格断点（累计 token >= min_tokens），并以 cap_tokens（创建上限，
/// 已含 85% 截断）为上界，保证 5m + 1h 之和等于 cache_creation_input_tokens。
fn compute_ttl_breakdown(
    profile: &PromptCacheProfile,
    matched_tokens: i32,
    cap_tokens: i32,
    min_tokens: i32,
) -> (i32, i32) {
    let mut cache_5m = 0i32;
    let mut cache_1h = 0i32;
    let mut previous = matched_tokens;
    for breakpoint in &profile.breakpoints {
        if breakpoint.cumulative_tokens < min_tokens {
            continue;
        }
        let current = breakpoint
            .cumulative_tokens
            .min(profile.total_input_tokens)
            .min(cap_tokens);
        if current <= previous {
            continue;
        }
        let delta = current - previous;
        if breakpoint.ttl >= ONE_HOUR {
            cache_1h += delta;
        } else {
            cache_5m += delta;
        }
        previous = current;
    }
    (cache_5m, cache_1h)
}

/// 计费口径下的 input_tokens：总输入减去缓存创建与缓存读取部分
pub fn billed_input_tokens(input_tokens: i32, cache: Option<&PromptCacheUsage>) -> i32 {
    match cache {
        Some(usage) => {
            (input_tokens - usage.cache_creation_input_tokens - usage.cache_read_input_tokens)
                .max(0)
        }
        None => input_tokens,
    }
}

/// 构建 usage JSON。携带缓存使用量时附加 cache 相关字段。
pub fn build_usage_json(
    input_tokens: i32,
    output_tokens: i32,
    cache: Option<&PromptCacheUsage>,
) -> Value {
    let mut usage = json!({
        "input_tokens": billed_input_tokens(input_tokens, cache),
        "output_tokens": output_tokens,
    });
    if let Some(c) = cache {
        usage["cache_creation_input_tokens"] = json!(c.cache_creation_input_tokens);
        usage["cache_read_input_tokens"] = json!(c.cache_read_input_tokens);
        usage["cache_creation"] = json!({
            "ephemeral_5m_input_tokens": c.cache_creation_5m_input_tokens,
            "ephemeral_1h_input_tokens": c.cache_creation_1h_input_tokens,
        });
    }
    usage
}

// === 请求扁平化为可缓存块序列 ===

struct CacheableBlock {
    canonical: String,
    tokens: i32,
    ttl: Duration,
    is_message_end: bool,
}

fn flatten_claude_cache_blocks(req: &MessagesRequest) -> Vec<CacheableBlock> {
    let mut blocks = Vec::new();

    // 请求前导：模型与 tool_choice 影响上游缓存键
    let prelude = json!({
        "kind": "request_prelude",
        "model": req.model,
        "tool_choice": req.tool_choice,
    });
    push_block(&mut blocks, &prelude, Duration::ZERO, false);

    if let Some(tools) = &req.tools {
        for tool in tools {
            let value = json!({
                "kind": "tool",
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            });
            let ttl = extract_cache_ttl_from_control(tool.cache_control.as_ref());
            push_block(&mut blocks, &value, ttl, false);
        }
    }

    if let Some(system) = &req.system {
        for msg in system {
            let block = json!({ "type": "text", "text": msg.text });
            if is_billing_header_block(&block) {
                continue;
            }
            let value = json!({ "kind": "system", "block": block });
            let ttl = extract_cache_ttl_from_control(msg.cache_control.as_ref());
            push_block(&mut blocks, &value, ttl, false);
        }
    }

    for msg in &req.messages {
        match &msg.content {
            Value::String(text) => {
                let block = json!({ "type": "text", "text": text });
                let value = json!({ "kind": "message", "role": msg.role, "block": block });
                push_block(&mut blocks, &value, Duration::ZERO, true);
            }
            Value::Array(items) => {
                let last_idx = items.len().saturating_sub(1);
                for (i, item) in items.iter().enumerate() {
                    if is_billing_header_block(item) {
                        continue;
                    }
                    let value = json!({ "kind": "message", "role": msg.role, "block": item });
                    let ttl = extract_cache_ttl(item);
                    push_block(&mut blocks, &value, ttl, i == last_idx);
                }
            }
            Value::Null => {}
            other => {
                let value = json!({ "kind": "message", "role": msg.role, "block": other });
                push_block(&mut blocks, &value, Duration::ZERO, true);
            }
        }
    }

    blocks
}

fn push_block(
    blocks: &mut Vec<CacheableBlock>,
    value: &Value,
    ttl: Duration,
    is_message_end: bool,
) {
    let canonical = canonicalize_cache_value(value);
    let tokens = estimate_approx_tokens(&canonical);
    blocks.push(CacheableBlock {
        canonical,
        tokens,
        ttl,
        is_message_end,
    });
}

/// Claude Code 的 x-anthropic-billing-header 系统块内容每请求漂移，
/// 且不影响模型语义，不参与指纹。
fn is_billing_header_block(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    // 只处理 text 块（或没有显式 type 但含 text 的块）
    if let Some(block_type) = obj.get("type").and_then(Value::as_str) {
        if !block_type.is_empty() && block_type != "text" {
            return false;
        }
    }
    let Some(text) = obj.get("text").and_then(Value::as_str) else {
        return false;
    };
    text.trim_start()
        .to_lowercase()
        .starts_with("x-anthropic-billing-header:")
}

/// 从内容块的 cache_control 字段提取 TTL（非 ephemeral 或缺失返回 0）
fn extract_cache_ttl(block: &Value) -> Duration {
    extract_cache_ttl_from_control(block.get("cache_control"))
}

fn extract_cache_ttl_from_control(cache_control: Option<&Value>) -> Duration {
    let Some(control) = cache_control.and_then(Value::as_object) else {
        return Duration::ZERO;
    };
    let cache_type = control.get("type").and_then(Value::as_str).unwrap_or("");
    if !cache_type.eq_ignore_ascii_case("ephemeral") {
        return Duration::ZERO;
    }
    let ttl = control
        .get("ttl")
        .and_then(parse_ttl_value)
        .unwrap_or(DEFAULT_PROMPT_CACHE_TTL);
    normalize_ttl(ttl)
}

fn parse_ttl_value(value: &Value) -> Option<Duration> {
    match value {
        Value::String(s) => {
            let trimmed = s.trim().to_lowercase();
            if trimmed.is_empty() {
                return None;
            }
            // 支持 "5m"、"1h"、"300s" 及纯数字秒
            if let Some(num) = trimmed.strip_suffix('h') {
                return num
                    .parse::<f64>()
                    .ok()
                    .filter(|v| *v > 0.0)
                    .map(|v| Duration::from_secs_f64(v * 3600.0));
            }
            if let Some(num) = trimmed.strip_suffix('m') {
                return num
                    .parse::<f64>()
                    .ok()
                    .filter(|v| *v > 0.0)
                    .map(|v| Duration::from_secs_f64(v * 60.0));
            }
            if let Some(num) = trimmed.strip_suffix('s') {
                return num
                    .parse::<f64>()
                    .ok()
                    .filter(|v| *v > 0.0)
                    .map(Duration::from_secs_f64);
            }
            trimmed
                .parse::<u64>()
                .ok()
                .filter(|v| *v > 0)
                .map(Duration::from_secs)
        }
        Value::Number(n) => n.as_f64().filter(|v| *v > 0.0).map(Duration::from_secs_f64),
        _ => None,
    }
}

/// TTL 归一化：Anthropic 只有 5m 和 1h 两档
fn normalize_ttl(ttl: Duration) -> Duration {
    if ttl.is_zero() {
        return Duration::ZERO;
    }
    if ttl > DEFAULT_PROMPT_CACHE_TTL {
        return ONE_HOUR;
    }
    DEFAULT_PROMPT_CACHE_TTL
}

/// 规范化 JSON 序列化：键排序、跳过任意层级的 cache_control 键，
/// 保证同一内容无论键序、是否带 cache_control 均得到同一指纹。
fn canonicalize_cache_value(value: &Value) -> String {
    let mut buf = String::new();
    write_canonical_json(&mut buf, value);
    buf
}

fn write_canonical_json(buf: &mut String, value: &Value) {
    match value {
        Value::Null => buf.push_str("null"),
        Value::Bool(b) => buf.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => buf.push_str(&n.to_string()),
        Value::String(s) => {
            buf.push_str(&serde_json::to_string(s).unwrap_or_default());
        }
        Value::Array(items) => {
            buf.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write_canonical_json(buf, item);
            }
            buf.push(']');
        }
        Value::Object(map) => {
            buf.push('{');
            let mut keys: Vec<&String> = map.keys().filter(|k| *k != "cache_control").collect();
            keys.sort();
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                buf.push_str(&serde_json::to_string(key).unwrap_or_default());
                buf.push(':');
                write_canonical_json(buf, &map[key.as_str()]);
            }
            buf.push('}');
        }
    }
}

/// 长度前缀分帧写入哈希，避免块边界歧义
fn write_hash_chunk(hasher: &mut Sha256, chunk: &str) {
    hasher.update(chunk.len().to_string().as_bytes());
    hasher.update([0u8]);
    hasher.update(chunk.as_bytes());
    hasher.update([0u8]);
}

/// 近似 token 估算（按字符类别加权，与 Kiro-Go 的 estimateApproxTokens 一致）
fn estimate_approx_tokens(text: &str) -> i32 {
    if text.is_empty() {
        return 0;
    }
    let length = text.chars().count();
    if length < 5 {
        return ((length as f64) / 3.0).ceil().max(1.0) as i32;
    }

    let mut regular_ascii = 0f64;
    let mut digits = 0f64;
    let mut symbols = 0f64;
    let mut non_ascii = 0f64;
    for c in text.chars() {
        if (c as u32) >= 0x80 {
            non_ascii += 1.0;
        } else if c.is_ascii_digit() {
            digits += 1.0;
        } else if matches!(c, '!'..='/' | ':'..='@' | '['..='`' | '{'..='~') {
            symbols += 1.0;
        } else {
            regular_ascii += 1.0;
        }
    }

    let estimated =
        (regular_ascii / 4.5 + digits / 2.0 + symbols / 1.5 + non_ascii / 1.5).ceil() as i32;
    estimated.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_from_json(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid MessagesRequest")
    }

    fn long_system_text() -> String {
        "You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. "
            .repeat(80)
    }

    #[test]
    fn test_compute_and_update() {
        let tracker = PromptCacheTracker::new();
        let req = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": [{
                "type": "text",
                "text": long_system_text(),
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "hello world"}]
        }));

        let mut profile = tracker
            .build_claude_profile(&req)
            .expect("profile should be built");
        profile.raise_total_input_tokens(120);

        let first = tracker.compute(1, &profile);
        assert!(
            first.cache_creation_input_tokens > 0,
            "first request should create cache tokens: {first:?}"
        );
        assert_eq!(first.cache_read_input_tokens, 0);

        tracker.update(1, &profile);
        let second = tracker.compute(1, &profile);
        assert!(
            second.cache_read_input_tokens > 0,
            "repeated request should read cache tokens: {second:?}"
        );
        assert_eq!(second.cache_creation_input_tokens, 0);
    }

    #[test]
    fn test_no_profile_without_cache_control() {
        let tracker = PromptCacheTracker::new();
        let req = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}]
        }));
        assert!(tracker.build_claude_profile(&req).is_none());
    }

    #[test]
    fn test_usage_json_includes_cache_fields() {
        let usage = PromptCacheUsage {
            cache_creation_input_tokens: 30,
            cache_read_input_tokens: 20,
            cache_creation_5m_input_tokens: 10,
            cache_creation_1h_input_tokens: 20,
        };

        let m = build_usage_json(100, 50, Some(&usage));
        assert_eq!(m["input_tokens"], 50);
        assert_eq!(m["cache_creation_input_tokens"], 30);
        assert_eq!(m["cache_read_input_tokens"], 20);
        assert_eq!(m["cache_creation"]["ephemeral_5m_input_tokens"], 10);
        assert_eq!(m["cache_creation"]["ephemeral_1h_input_tokens"], 20);

        let plain = build_usage_json(100, 50, None);
        assert_eq!(plain["input_tokens"], 100);
        assert!(plain.get("cache_read_input_tokens").is_none());
    }

    #[test]
    fn test_stable_across_billing_header_drift() {
        let tracker = PromptCacheTracker::new();
        let build = |billing_header: &str| {
            request_from_json(json!({
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "system": [
                    {"type": "text", "text": billing_header},
                    {
                        "type": "text",
                        "text": long_system_text(),
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": "hello world"}]
            }))
        };

        let req1 = build("x-anthropic-billing-header: cc_version=2.1.87.1; cch=aaaa;");
        let mut profile1 = tracker.build_claude_profile(&req1).unwrap();
        profile1.raise_total_input_tokens(2048);
        let first = tracker.compute(1, &profile1);
        assert_eq!(first.cache_read_input_tokens, 0);
        tracker.update(1, &profile1);

        let req2 =
            build("x-anthropic-billing-header: cc_version=2.1.87.42; cch=bbbb; padding=xxyyzz;");
        let mut profile2 = tracker.build_claude_profile(&req2).unwrap();
        profile2.raise_total_input_tokens(2048);
        let second = tracker.compute(1, &profile2);
        assert!(
            second.cache_read_input_tokens > 0,
            "billing header drift should not break cache hits: {second:?}"
        );
    }

    #[test]
    fn test_stable_when_billing_header_appears_or_disappears() {
        let tracker = PromptCacheTracker::new();
        let build = |include_billing: bool| {
            let mut system = Vec::new();
            if include_billing {
                system.push(json!({
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.87.1; cch=aaaa;"
                }));
            }
            system.push(json!({
                "type": "text",
                "text": long_system_text(),
                "cache_control": {"type": "ephemeral"}
            }));
            request_from_json(json!({
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "system": system,
                "messages": [{"role": "user", "content": "hello world"}]
            }))
        };

        let mut with_billing = tracker.build_claude_profile(&build(true)).unwrap();
        with_billing.raise_total_input_tokens(2048);
        tracker.update(1, &with_billing);

        let mut without_billing = tracker.build_claude_profile(&build(false)).unwrap();
        without_billing.raise_total_input_tokens(2048);
        let result = tracker.compute(1, &without_billing);
        assert!(
            result.cache_read_input_tokens > 0,
            "cache should hit when billing header disappears: {result:?}"
        );
    }

    #[test]
    fn test_implicit_breakpoint_at_message_end() {
        let tracker = PromptCacheTracker::new();
        let system = json!([{
            "type": "text",
            "text": long_system_text(),
            "cache_control": {"type": "ephemeral"}
        }]);

        // 第一轮：单条用户消息
        let req1 = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": system,
            "messages": [{"role": "user", "content": "question one"}]
        }));
        let mut profile1 = tracker.build_claude_profile(&req1).unwrap();
        profile1.raise_total_input_tokens(2048);
        tracker.update(1, &profile1);

        // 第二轮：对话继续。最新消息没有显式 cache_control，
        // 应通过隐式消息尾断点命中已存储的前缀。
        let req2 = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": system,
            "messages": [
                {"role": "user", "content": "question one"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "follow-up question"}
            ]
        }));
        let mut profile2 = tracker.build_claude_profile(&req2).unwrap();
        profile2.raise_total_input_tokens(4096);
        let result = tracker.compute(1, &profile2);
        assert!(
            result.cache_read_input_tokens > 0,
            "implicit message-end breakpoint should hit cache: {result:?}"
        );
    }

    #[test]
    fn test_accounts_are_isolated() {
        let tracker = PromptCacheTracker::new();
        let req = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": [{
                "type": "text",
                "text": long_system_text(),
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "hello world"}]
        }));
        let mut profile = tracker.build_claude_profile(&req).unwrap();
        profile.raise_total_input_tokens(2048);

        tracker.update(1, &profile);
        let other_account = tracker.compute(2, &profile);
        assert_eq!(
            other_account.cache_read_input_tokens, 0,
            "cache entries must not leak across credentials"
        );
    }

    #[test]
    fn test_min_cacheable_threshold() {
        let tracker = PromptCacheTracker::new();
        // 短 system：低于 1024 token 阈值，不应报告缓存创建（含 5m/1h 明细）
        let req = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": [{
                "type": "text",
                "text": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let profile = tracker.build_claude_profile(&req).unwrap();
        let usage = tracker.compute(1, &profile);
        assert_eq!(usage, PromptCacheUsage::default());

        // opus 阈值更高（4096）：1024 级别的前缀也不可缓存
        let opus_req = request_from_json(json!({
            "model": "claude-opus-4-8",
            "max_tokens": 100,
            "system": [{
                "type": "text",
                "text": "word ".repeat(1200),
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let opus_profile = tracker.build_claude_profile(&opus_req).unwrap();
        let opus_usage = tracker.compute(1, &opus_profile);
        assert_eq!(opus_usage, PromptCacheUsage::default());
    }

    #[test]
    fn test_first_request_creation_is_ratio_capped() {
        let tracker = PromptCacheTracker::new();
        let profile = PromptCacheProfile {
            breakpoints: vec![PromptCacheBreakpoint {
                fingerprint: [1u8; 32],
                cumulative_tokens: 10_000,
                ttl: DEFAULT_PROMPT_CACHE_TTL,
            }],
            total_input_tokens: 10_000,
            model: "claude-sonnet-4-6".to_string(),
        };

        let usage = tracker.compute(1, &profile);
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 8_500);
        assert_eq!(usage.cache_creation_5m_input_tokens, 8_500);
        assert_eq!(usage.cache_creation_1h_input_tokens, 0);
    }

    #[test]
    fn test_profile_is_capped_to_model_context_window() {
        let tracker = PromptCacheTracker::new();
        let mut profile = PromptCacheProfile {
            breakpoints: vec![PromptCacheBreakpoint {
                fingerprint: [2u8; 32],
                cumulative_tokens: 5_033_520,
                ttl: DEFAULT_PROMPT_CACHE_TTL,
            }],
            total_input_tokens: 5_033_520,
            model: "claude-opus-4-7".to_string(),
        };
        profile.cap_to_context_window();

        assert_eq!(profile.total_input_tokens, 1_000_000);
        assert_eq!(profile.breakpoints[0].cumulative_tokens, 1_000_000);

        let usage = tracker.compute(1, &profile);
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 850_000);
        assert_eq!(usage.cache_creation_5m_input_tokens, 850_000);
        assert_eq!(usage.cache_creation_1h_input_tokens, 0);
    }

    /// 不变式：任何路径下 5m + 1h 明细之和必须等于 cache_creation_input_tokens。
    /// sub2api 按明细独立计费（computeCacheCreationCost），二者不一致会导致
    /// "聚合为 0 但明细仍计费" 的口径错误。
    #[test]
    fn test_ttl_breakdown_matches_aggregate() {
        let tracker = PromptCacheTracker::new();
        let req = request_from_json(json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": [{
                "type": "text",
                "text": long_system_text(),
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [
                {"role": "user", "content": "question one"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": long_system_text()}
            ]
        }));

        // 首次请求（无命中）
        let mut profile = tracker.build_claude_profile(&req).unwrap();
        profile.raise_total_input_tokens(4096);
        let first = tracker.compute(1, &profile);
        assert_eq!(
            first.cache_creation_5m_input_tokens + first.cache_creation_1h_input_tokens,
            first.cache_creation_input_tokens,
            "first request breakdown mismatch: {first:?}"
        );

        // 命中后（部分创建 + 部分命中，触发 85% 截断路径）
        tracker.update(1, &profile);
        let second = tracker.compute(1, &profile);
        assert_eq!(
            second.cache_creation_5m_input_tokens + second.cache_creation_1h_input_tokens,
            second.cache_creation_input_tokens,
            "hit-path breakdown mismatch: {second:?}"
        );
    }

    #[test]
    fn test_ttl_normalization_and_breakdown() {
        assert_eq!(normalize_ttl(Duration::ZERO), Duration::ZERO);
        assert_eq!(
            normalize_ttl(Duration::from_secs(60)),
            DEFAULT_PROMPT_CACHE_TTL
        );
        assert_eq!(
            normalize_ttl(Duration::from_secs(300)),
            DEFAULT_PROMPT_CACHE_TTL
        );
        assert_eq!(normalize_ttl(Duration::from_secs(301)), ONE_HOUR);
        assert_eq!(normalize_ttl(Duration::from_secs(7200)), ONE_HOUR);

        assert_eq!(
            parse_ttl_value(&json!("5m")),
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            parse_ttl_value(&json!("1h")),
            Some(Duration::from_secs(3600))
        );
        assert_eq!(parse_ttl_value(&json!(300)), Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_canonicalization_ignores_cache_control_and_key_order() {
        let with_control = json!({
            "type": "text",
            "text": "stable",
            "cache_control": {"type": "ephemeral"}
        });
        let without_control = json!({
            "text": "stable",
            "type": "text"
        });
        assert_eq!(
            canonicalize_cache_value(&with_control),
            canonicalize_cache_value(&without_control)
        );
    }
}
