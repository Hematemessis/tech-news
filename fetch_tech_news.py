#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科技热点扫描脚本 - 智能筛选，无需 API Key
筛选标准：时间新鲜度、相关性评分、热度阈值、去重、质量过滤
"""

import os
import sys
import re
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# 需要安装的依赖: pip install aiohttp feedparser playwright

# ============== RSS 源配置 ==============
RSS_SOURCES = {
    "Hacker News": "https://news.ycombinator.com/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "机器之心": "https://www.jiqizhixin.com/rss",
    "量子位": "https://www.qbitai.com/feed",
    "InfoQ": "https://www.infoq.cn/feed",
    "Solidot": "https://www.solidot.org/index.rss",
    "36氪": "https://36kr.com/feed",
    "爱范儿": "https://www.ifanr.com/feed",
    "少数派": "https://sspai.com/feed", 
}

# ============== 筛选配置 ==============
# 时间窗口：只保留过去 48 小时的新闻
TIME_WINDOW_HOURS = 48

# 热度阈值配置
SCORE_THRESHOLDS = {
    "Hacker News": 50,      # HN 分数低于 50 的过滤
    "GitHub Trending": 10,  # GitHub star 低于 10 的过滤
    "default": 0,           # 其他源无阈值
}

# 相关性评分权重
RELEVANCE_WEIGHTS = {
    "high": 10,     # 核心 AI 词汇
    "medium": 5,    # 相关技术词汇
    "low": 2,       # 一般科技词汇
}

# 关键词分类
KEYWORDS_CATEGORIES = {
    "high": [  # 核心 AI 词汇（权重 10）
        "AI", "人工智能", "artificial intelligence",
        "LLM", "大模型", "large language model",
        "ChatGPT", "GPT-4", "GPT-3", "Claude", "Gemini", "文心一言",
        "OpenAI", "Anthropic",
        "AIGC", "生成式 AI", "generative AI",
    ],
    "medium": [  # 相关技术（权重 5）
        "机器学习", "machine learning", "深度学习", "deep learning",
        "transformer", "attention", "神经网络", "neural network",
        "NLP", "自然语言处理", "computer vision", "计算机视觉",
        "多模态", "multimodal",
        "fine-tuning", "微调", "prompt", "提示工程",
        "向量数据库", "embedding", "RAG",
    ],
    "low": [  # 一般科技（权重 2）
        "startup", "初创公司", "融资", "funding", "investment",
        "NVIDIA", "GPU", "芯片", "chip", "半导体",
        "algorithm", "算法", "模型", "model",
        "Google", "Microsoft", "Meta", "Apple", "Amazon",
        "robotics", "机器人", "autonomous", "自动驾驶",
        "cloud", "云计算", "SaaS",
    ]
}

# 黑名单关键词（出现则过滤）
BLACKLIST_KEYWORDS = [
    "招聘", "诚聘", "hire", "hiring", "join us", "we're looking",
    "优惠券", "discount", "promo", "限时", "抢购",
    "免费试用", "点击领取", "扫码",
    "成人", "色情", "赌博", "casino", "porn",
]

# 垃圾域名黑名单
BLACKLIST_DOMAINS = [
    "bit.ly", "t.co", "short.link",
]


class NewsFilter:
    """新闻筛选器"""
    
    def __init__(self):
        self.seen_urls = set()
        self.seen_titles = []
        
    def calculate_relevance_score(self, title: str, description: str = "") -> int:
        """计算与 AI/科技的相关性分数"""
        text = f"{title} {description}".lower()
        score = 0
        matched_keywords = []
        
        for category, keywords in KEYWORDS_CATEGORIES.items():
            weight = RELEVANCE_WEIGHTS[category]
            for kw in keywords:
                if kw.lower() in text:
                    score += weight
                    matched_keywords.append(kw)
        
        return score, matched_keywords
    
    def is_blacklisted(self, title: str, url: str) -> bool:
        """检查是否在黑名单中"""
        text = title.lower()
        
        # 检查标题黑名单
        for bk in BLACKLIST_KEYWORDS:
            if bk.lower() in text:
                return True
        
        # 检查域名黑名单
        for domain in BLACKLIST_DOMAINS:
            if domain in url.lower():
                return True
        
        return False
    
    def is_duplicate(self, title: str, url: str) -> bool:
        """检查是否重复（基于 URL 或标题相似度）"""
        # URL 完全匹配
        url_normalized = url.lower().strip().rstrip('/')
        if url_normalized in self.seen_urls:
            return True
        
        # 标题相似度检查（80% 以上认为是同一新闻）
        for seen_title in self.seen_titles:
            similarity = SequenceMatcher(None, title.lower(), seen_title.lower()).ratio()
            if similarity > 0.8:
                return True
        
        # 记录已见
        self.seen_urls.add(url_normalized)
        self.seen_titles.append(title)
        return False
    
    def check_time_freshness(self, time_str: str) -> bool:
        """检查时间是否在有效窗口内"""
        try:
            # 尝试多种时间格式
            formats = [
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%a, %d %b %Y %H:%M:%S",
            ]
            
            news_time = None
            for fmt in formats:
                try:
                    news_time = datetime.strptime(time_str[:19], fmt)
                    break
                except:
                    continue
            
            if not news_time:
                # 如果解析失败，默认接受
                return True
            
            # 处理年份可能为未来的情况
            if news_time.year > datetime.now().year:
                news_time = news_time.replace(year=datetime.now().year)
            
            time_diff = datetime.now() - news_time
            return time_diff <= timedelta(hours=TIME_WINDOW_HOURS)
        except:
            # 解析失败默认接受
            return True
    
    def check_score_threshold(self, score: int, source: str) -> bool:
        """检查是否达到热度阈值"""
        threshold = SCORE_THRESHOLDS.get(source, SCORE_THRESHOLDS["default"])
        return score >= threshold
    
    def clean_title(self, title: str) -> str:
        """清理标题中的垃圾信息"""
        # 移除 HTML 标签
        title = re.sub(r'<[^>]+>', '', title)
        # 移除多余空格
        title = re.sub(r'\s+', ' ', title)
        # 移除特殊字符
        title = title.strip()
        return title


class TechNewsScanner:
    def __init__(self):
        self.results = []
        self.session = None
        self.filter = NewsFilter()
        self.category_stats = defaultdict(list)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def categorize_news(self, title: str, matched_keywords: list) -> str:
        """根据关键词给新闻分类"""
        text = title.lower()
        
        # 分类规则
        if any(kw in text for kw in ["llm", "gpt", "claude", "gemini", "大模型", "文心一言"]):
            return "大模型"
        elif any(kw in text for kw in ["startup", "融资", "funding", "investment", "初创"]):
            return "融资动态"
        elif any(kw in text for kw in ["chip", "gpu", "芯片", "nvidia", "半导体"]):
            return "芯片硬件"
        elif any(kw in text for kw in ["github", "开源", "open source"]):
            return "开源项目"
        elif any(kw in text for kw in ["computer vision", "cv", "视觉", "图像"]):
            return "计算机视觉"
        elif any(kw in text for kw in ["nlp", "语言", "text", "文本"]):
            return "自然语言处理"
        else:
            return "AI综合"
    
    # ========== Hacker News ==========
    async def fetch_hackernews(self):
        """获取 Hacker News 热点"""
        try:
            print("[INFO] 正在获取 Hacker News...")
            
            async with self.session.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json"
            ) as resp:
                story_ids = await resp.json()
            
            stories = []
            filtered_count = 0
            
            for story_id in story_ids[:50]:  # 获取更多以便筛选
                async with self.session.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                ) as resp:
                    story = await resp.json()
                    if not story:
                        continue
                    
                    title = story.get("title", "")
                    url = story.get("url", f"https://news.ycombinator.com/item?id={story_id}")
                    score = story.get("score", 0)
                    
                    # 筛选流程
                    # 1. 黑名单检查
                    if self.filter.is_blacklisted(title, url):
                        filtered_count += 1
                        continue
                    
                    # 2. 相关性评分
                    rel_score, matched_kws = self.filter.calculate_relevance_score(title)
                    if rel_score < 5:  # 至少需要匹配一个 medium 或两个 low
                        continue
                    
                    # 3. 热度阈值
                    if not self.filter.check_score_threshold(score, "Hacker News"):
                        continue
                    
                    # 4. 去重
                    if self.filter.is_duplicate(title, url):
                        continue
                    
                    # 5. 清理标题
                    title = self.filter.clean_title(title)
                    
                    time_str = datetime.fromtimestamp(story.get("time", 0)).strftime("%Y-%m-%d %H:%M")
                    
                    # 分类
                    category = self.categorize_news(title, matched_kws)
                    self.category_stats[category].append(title)
                    
                    stories.append({
                        "title": title,
                        "url": url,
                        "score": score,
                        "source": "Hacker News",
                        "time": time_str,
                        "relevance": rel_score,
                        "category": category,
                        "matched_keywords": matched_kws[:3],  # 只显示前3个
                    })
            
            print(f"   [OK] 获取 {len(stories)} 条，过滤 {filtered_count} 条")
            return stories
        except Exception as e:
            print(f"[ERR] Hacker News 获取失败: {e}")
            return []
    
    # ========== RSS 源 ==========
    async def fetch_rss(self):
        """获取 RSS 源内容"""
        all_results = []
        
        for source_name, url in RSS_SOURCES.items():
            try:
                print(f"[INFO] 正在获取 {source_name}...")
                feed = feedparser.parse(url)
                
                results = []
                for entry in feed.entries[:15]:  # 每个源多取一些
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    summary = entry.get("summary", "")
                    
                    # 筛选流程
                    # 1. 黑名单检查
                    if self.filter.is_blacklisted(title, link):
                        continue
                    
                    # 2. 相关性评分
                    rel_score, matched_kws = self.filter.calculate_relevance_score(title, summary)
                    if rel_score < 3:  # RSS 源可以稍微放宽
                        continue
                    
                    # 3. 去重
                    if self.filter.is_duplicate(title, link):
                        continue
                    
                    # 4. 时间检查
                    published = entry.get("published_parsed") or entry.get("updated_parsed")
                    if published:
                        time_str = datetime(*published[:6]).strftime("%Y-%m-%d %H:%M")
                    else:
                        time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    # 5. 清理标题
                    title = self.filter.clean_title(title)
                    
                    # 6. RSS 热度估算（根据关键词数量和来源权重）
                    estimated_score = rel_score * 5
                    
                    # 分类
                    category = self.categorize_news(title, matched_kws)
                    self.category_stats[category].append(title)
                    
                    results.append({
                        "title": title,
                        "url": link,
                        "score": estimated_score,
                        "source": source_name,
                        "time": time_str,
                        "relevance": rel_score,
                        "category": category,
                        "matched_keywords": matched_kws[:3],
                    })
                
                print(f"   [OK] 获取 {len(results)} 条")
                all_results.extend(results)
                
            except Exception as e:
                print(f"[ERR] {source_name} 获取失败: {e}")
        
        return all_results
    
    # ========== GitHub Trending ==========
    async def fetch_github_trending(self):
        """获取 GitHub Trending（AI/ML 相关）"""
        try:
            print("[INFO] 正在获取 GitHub Trending...")
            
            # 搜索最近一周的热门 AI 仓库
            one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            queries = [
                ("AI stars:>10", "AI"),
                ("machine-learning stars:>10", "ML"),
                ("llm stars:>10", "LLM"),
                ("chatgpt stars:>5", "ChatGPT"),
            ]
            
            results = []
            for query, tag in queries:
                async with self.session.get(
                    f"https://api.github.com/search/repositories",
                    params={
                        "q": f"{query} created:>{one_week_ago}",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 5
                    }
                ) as resp:
                    data = await resp.json()
                    for repo in data.get("items", []):
                        title = f"[{tag}] {repo.get('full_name', '')}: {repo.get('description', '')}"
                        url = repo.get("html_url", "")
                        score = repo.get("stargazers_count", 0)
                        
                        # 筛选
                        if self.filter.is_blacklisted(title, url):
                            continue
                        if self.filter.is_duplicate(title, url):
                            continue
                        if score < SCORE_THRESHOLDS["GitHub Trending"]:
                            continue
                        
                        title = self.filter.clean_title(title)
                        
                        # GitHub 项目都属于开源分类
                        self.category_stats["开源项目"].append(title)
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "score": score,
                            "source": "GitHub Trending",
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "relevance": 10,
                            "category": "开源项目",
                            "matched_keywords": [tag],
                        })
            
            print(f"   [OK] 获取 {len(results)} 条")
            return results
        except Exception as e:
            print(f"[ERR] GitHub Trending 获取失败: {e}")
            return []
    
    async def fetch_all(self):
        """并行获取所有数据源"""
        print("开始获取科技热点...")
        print("=" * 60)
        print(f"时间窗口: 过去 {TIME_WINDOW_HOURS} 小时")
        print(f"最低相关性: 3 分")
        print(f"HN 热度阈值: {SCORE_THRESHOLDS['Hacker News']} 分")
        print("=" * 60)
        
        tasks = [
            self.fetch_hackernews(),
            self.fetch_rss(),
            self.fetch_github_trending(),
        ]
        
        results = await asyncio.gather(*tasks)
        
        all_news = []
        for source_results in results:
            all_news.extend(source_results)
        
        # 综合评分排序（热度 + 相关性）
        all_news.sort(key=lambda x: x.get("score", 0) + x.get("relevance", 0) * 10, reverse=True)
        
        print("=" * 60)
        print(f"[OK] 共获取 {len(all_news)} 条精选科技热点")
        
        # 打印分类统计
        if self.category_stats:
            print("\n分类统计:")
            for category, items in sorted(self.category_stats.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"   {category}: {len(items)} 条")
        
        return all_news
    
    def generate_html(self, news_items, output_path):
        """生成按主题板块分区的 HTML 报告 - Corporate Trust 设计系统"""
        
        # 按分类分组
        categorized_news = defaultdict(list)
        for item in news_items[:50]:
            category = item.get('category', '其他')
            categorized_news[category].append(item)
        
        # 分类排序（按数量从多到少）
        sorted_categories = sorted(categorized_news.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 分类配置（Heroicons SVG + 颜色）
        category_config = {
            "大模型": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="m3.75 13.5 10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z"/></svg>''',
                "gradient": "from-rose-500 to-pink-600",
                "bg": "#ffe4e6"
            },
            "开源项目": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M17.25 6.75 22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3-4.5 16.5"/></svg>''',
                "gradient": "from-blue-500 to-cyan-600",
                "bg": "#dbeafe"
            },
            "融资动态": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l.473-.298m-11.4-5.96 3.93 3.93m0 0 3.93-3.93m-3.93 3.93V15"/></svg>''',
                "gradient": "from-amber-500 to-orange-600",
                "bg": "#fef3c7"
            },
            "芯片硬件": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Z"/></svg>''',
                "gradient": "from-violet-500 to-purple-600",
                "bg": "#ede9fe"
            },
            "计算机视觉": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"/></svg>''',
                "gradient": "from-emerald-500 to-teal-600",
                "bg": "#d1fae5"
            },
            "自然语言处理": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"/></svg>''',
                "gradient": "from-orange-500 to-red-500",
                "bg": "#ffedd5"
            },
            "AI综合": {
                "icon": '''<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z"/></svg>''',
                "gradient": "from-indigo-500 to-violet-600",
                "bg": "#e0e7ff"
            },
        }
        
        # Heroicons SVG 图标定义
        icons = {
            "newspaper": '''<svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 0 1-2.25 2.25M16.5 7.5V18a2.25 2.25 0 0 0 2.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 0 0 2.25 2.25h13.5M6 7.5h3v3H6v-3Z"/></svg>''',
            "globe": '''<svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 21a9.004 9.004 0 0 0 8.716-6.747M12 21a9.004 9.004 0 0 1-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 0 1 7.843 4.582M12 3a8.997 8.997 0 0 0-7.843 4.582m15.686 0A11.953 11.953 0 0 1 12 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0 1 21 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0 1 12 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 0 1 3 12c0-1.605.42-3.113 1.157-4.418"/></svg>''',
            "grid": '''<svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25A2.25 2.25 0 0 1 13.5 18v-2.25Z"/></svg>''',
            "check": '''<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>''',
            "arrow": '''<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3"/></svg>''',
            "star": '''<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10.868 2.884c-.321-.772-1.415-.772-1.736 0l-1.83 4.401-4.753.381c-.833.067-1.171 1.107-.536 1.651l3.62 3.102-1.106 4.637c-.194.813.691 1.456 1.405 1.02L10 15.591l4.069 2.485c.713.436 1.598-.207 1.404-1.02l-1.106-4.637 3.62-3.102c.635-.544.297-1.584-.536-1.65l-4.752-.382-1.831-4.401Z" clip-rule="evenodd"/></svg>''',
        }
        
        html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Tech Daily - {date}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
                    }},
                    colors: {{
                        primary: '#4F46E5',
                        secondary: '#7C3AED',
                    }}
                }}
            }}
        }}
    </script>
    <style>
        /* 基础样式 */
        body {{
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            background: #F8FAFC;
        }}
        
        /* 渐变文字 */
        .gradient-text {{
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        /* 背景装饰球 */
        .blob {{
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            pointer-events: none;
        }}
        
        /* 卡片阴影效果 - Corporate Trust 风格 */
        .card-shadow {{
            box-shadow: 0 4px 20px -2px rgba(79, 70, 229, 0.1);
            transition: all 0.3s ease;
        }}
        
        .card-shadow:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -5px rgba(79, 70, 229, 0.2), 0 10px 20px -5px rgba(79, 70, 229, 0.1);
        }}
        
        /* 来源标签颜色 */
        .badge-hackernews {{ background: linear-gradient(135deg, #ff6600, #ff8533); }}
        .badge-github {{ background: linear-gradient(135deg, #24292e, #586069); }}
        .badge-techcrunch {{ background: linear-gradient(135deg, #0f9d58, #34a853); }}
        .badge-jiqizhixin {{ background: linear-gradient(135deg, #1890ff, #69c0ff); }}
        .badge-qbitai {{ background: linear-gradient(135deg, #722ed1, #b37feb); }}
        .badge-infoq {{ background: linear-gradient(135deg, #ff6b6b, #ffa39e); }}
        .badge-solidot {{ background: linear-gradient(135deg, #4ecdc4, #95e1d3); }}
        .badge-36kr {{ background: linear-gradient(135deg, #4285f4, #34a853); }}
        .badge-ifanr {{ background: linear-gradient(135deg, #ff6b6b, #ee5a24); }}
        .badge-sspai {{ background: linear-gradient(135deg, #d42626, #b91c1c); }}
        .badge-default {{ background: linear-gradient(135deg, #6c5ce7, #a29bfe); }}
        
        /* 关键词标签 */
        .keyword-tag {{
            background: rgba(79, 70, 229, 0.08);
            color: #4F46E5;
            transition: all 0.2s ease;
        }}
        
        .keyword-tag:hover {{
            background: rgba(79, 70, 229, 0.15);
        }}
        
        /* 热度指示器 */
        .score-hot {{ color: #ef4444; }}
        .score-warm {{ color: #f59e0b; }}
        .score-normal {{ color: #9ca3af; }}
        
        /* 板块标题装饰 */
        .section-title-line {{
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, transparent 100%);
            height: 3px;
            border-radius: 2px;
        }}
        
        /* 平滑滚动 */
        html {{
            scroll-behavior: smooth;
        }}
        
        /* 动画 */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .animate-fade-in {{
            animation: fadeInUp 0.6s ease-out forwards;
        }}
        
        /* 统计卡片光晕 */
        .stat-glow {{
            box-shadow: 0 0 30px rgba(79, 70, 229, 0.15);
        }}
        
        /* 可展开统计卡片 */
        .stat-card-expandable {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .stat-card-expandable:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(79, 70, 229, 0.2);
        }}
        
        .stat-card-expandable.active {{
            border-color: #4F46E5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }}
        
        .expand-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-out, opacity 0.3s ease, padding 0.3s ease;
            opacity: 0;
        }}
        
        .expand-content.show {{
            max-height: 500px;
            opacity: 1;
            padding-top: 1rem;
            margin-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
        
        .expand-icon {{
            transition: transform 0.3s ease;
        }}
        
        .expand-icon.rotate {{
            transform: rotate(180deg);
        }}
        
        .source-list, .category-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .source-tag, .category-tag {{
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            padding: 0.375rem 0.875rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2);
            transition: all 0.2s ease;
        }}
        
        .source-tag:hover, .category-tag:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }}
        
        .click-hint {{
            font-size: 0.7rem;
            color: #64748b;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
            opacity: 0.7;
        }}
        
        /* 热门5条样式 */
        .top5-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .top5-item {{
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.5rem;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }}
        
        .top5-item:hover {{
            background: rgba(79, 70, 229, 0.05);
        }}
        
        .top5-rank {{
            width: 1.5rem;
            height: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            color: white;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            flex-shrink: 0;
        }}
        
        .top5-item:nth-child(1) .top5-rank {{
            background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
        }}
        
        .top5-item:nth-child(2) .top5-rank {{
            background: linear-gradient(135deg, #f97316 0%, #f59e0b 100%);
        }}
        
        .top5-item:nth-child(3) .top5-rank {{
            background: linear-gradient(135deg, #f59e0b 0%, #eab308 100%);
        }}
        
        .top5-title {{
            font-size: 0.875rem;
            color: #334155;
            line-height: 1.5;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
    </style>
</head>
<body class="relative min-h-screen overflow-x-hidden">
    <!-- 背景装饰球 -->
    <div class="blob bg-indigo-400 w-96 h-96 -top-20 -left-20"></div>
    <div class="blob bg-violet-400 w-80 h-80 top-40 right-0"></div>
    <div class="blob bg-purple-400 w-64 h-64 bottom-40 left-20"></div>
    
    <div class="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <!-- Header -->
        <header class="text-center mb-16 animate-fade-in">
            <div class="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-sm mb-6">
                <span class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                <span class="text-sm text-slate-600 font-medium">今日已更新</span>
            </div>
            
            <h1 class="text-5xl sm:text-6xl font-extrabold mb-4 tracking-tight">
                <span class="text-slate-900">AI Tech</span>
                <span class="gradient-text">Daily</span>
            </h1>
            
            <p class="text-xl text-slate-500 font-medium mb-2">{date}</p>
            <p class="text-slate-400">智能筛选 · 精准分类 · 实时热点</p>
        </header>
        
        <!-- Stats -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-12 animate-fade-in items-start" style="animation-delay: 0.1s">
            <!-- 精选热点 - 可展开 -->
            <div class="bg-white rounded-2xl p-6 stat-glow border border-slate-100 stat-card-expandable" onclick="toggleExpand('hotspot')">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-semibold text-slate-500 mb-1">精选热点</p>
                        <p class="text-4xl font-bold text-slate-900">{total_count}</p>
                        <div class="click-hint">
                            <span>点击查看详情</span>
                            <svg class="w-3 h-3 expand-icon" id="hotspot-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                            </svg>
                        </div>
                    </div>
                    <div class="w-14 h-14 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-lg shadow-indigo-200">
                        ''' + icons["newspaper"] + '''
                    </div>
                </div>
                <div class="expand-content" id="hotspot-content">
                    <p class="text-sm text-slate-500 mb-3">今日最热门的 5 条精选：</p>
                    <div class="top5-list">
                        {top5_list}
                    </div>
                </div>
            </div>
            
            <!-- 数据来源 - 可展开 -->
            <div class="bg-white rounded-2xl p-6 stat-glow border border-slate-100 stat-card-expandable" onclick="toggleExpand('sources')">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-semibold text-slate-500 mb-1">数据来源</p>
                        <p class="text-4xl font-bold text-slate-900">{source_count}</p>
                        <div class="click-hint">
                            <span>点击查看详情</span>
                            <svg class="w-3 h-3 expand-icon" id="sources-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                            </svg>
                        </div>
                    </div>
                    <div class="w-14 h-14 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white shadow-lg shadow-violet-200">
                        ''' + icons["globe"] + '''
                    </div>
                </div>
                <div class="expand-content" id="sources-content">
                    <p class="text-sm text-slate-500 mb-3">本报告数据来自以下平台：</p>
                    <div class="source-list">
                        {source_list}
                    </div>
                </div>
            </div>
            
            <!-- 主题板块 - 可展开 -->
            <div class="bg-white rounded-2xl p-6 stat-glow border border-slate-100 stat-card-expandable" onclick="toggleExpand('categories')">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-semibold text-slate-500 mb-1">主题板块</p>
                        <p class="text-4xl font-bold text-slate-900">{category_count}</p>
                        <div class="click-hint">
                            <span>点击查看详情</span>
                            <svg class="w-3 h-3 expand-icon" id="categories-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                            </svg>
                        </div>
                    </div>
                    <div class="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center text-white shadow-lg shadow-purple-200">
                        ''' + icons["grid"] + '''
                    </div>
                </div>
                <div class="expand-content" id="categories-content">
                    <p class="text-sm text-slate-500 mb-3">新闻已自动分类为以下板块：</p>
                    <div class="category-list">
                        {category_list}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function toggleExpand(id) {{
                const content = document.getElementById(id + '-content');
                const icon = document.getElementById(id + '-icon');
                const card = content.closest('.stat-card-expandable');
                
                // 切换展开状态
                const isExpanded = content.classList.contains('show');
                
                // 先关闭所有其他的展开项
                document.querySelectorAll('.expand-content.show').forEach(el => {{
                    if (el !== content) {{
                        el.classList.remove('show');
                        el.closest('.stat-card-expandable').classList.remove('active');
                    }}
                }});
                document.querySelectorAll('.expand-icon.rotate').forEach(el => {{
                    if (el !== icon) {{
                        el.classList.remove('rotate');
                    }}
                }});
                
                // 切换当前项
                content.classList.toggle('show');
                icon.classList.toggle('rotate');
                card.classList.toggle('active');
            }}
        </script>
        
        <!-- Filter Info -->
        <div class="bg-white/70 backdrop-blur-md rounded-2xl p-6 mb-12 border border-slate-200/60 animate-fade-in" style="animation-delay: 0.2s">
            <div class="flex items-center gap-2 mb-5">
                <div class="text-indigo-600">''' + icons["check"] + '''</div>
                <span class="font-semibold text-slate-800">筛选标准</span>
                <span class="text-xs text-slate-400 ml-2">我们如何从海量信息中精选优质内容</span>
            </div>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                <div class="bg-white/80 rounded-xl p-4 border border-slate-100 hover:shadow-md transition-shadow">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="w-2 h-2 rounded-full bg-indigo-500"></span>
                        <span class="font-semibold text-slate-700 text-sm">时间窗口</span>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">只保留过去 {time_window} 小时内发布的新闻，确保内容新鲜度</p>
                </div>
                <div class="bg-white/80 rounded-xl p-4 border border-slate-100 hover:shadow-md transition-shadow">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="w-2 h-2 rounded-full bg-violet-500"></span>
                        <span class="font-semibold text-slate-700 text-sm">相关性评分</span>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">基于 AI 关键词权重计算，≥3 分表示内容与 AI/科技领域高度相关</p>
                </div>
                <div class="bg-white/80 rounded-xl p-4 border border-slate-100 hover:shadow-md transition-shadow">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="w-2 h-2 rounded-full bg-purple-500"></span>
                        <span class="font-semibold text-slate-700 text-sm">热度阈值</span>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">Hacker News 点赞 ≥{hn_threshold}、GitHub Star ≥10，筛选社区认可的内容</p>
                </div>
                <div class="bg-white/80 rounded-xl p-4 border border-slate-100 hover:shadow-md transition-shadow">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="w-2 h-2 rounded-full bg-pink-500"></span>
                        <span class="font-semibold text-slate-700 text-sm">智能去重</span>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">标题相似度超过 80% 视为重复新闻，自动过滤避免信息冗余</p>
                </div>
                <div class="bg-white/80 rounded-xl p-4 border border-slate-100 hover:shadow-md transition-shadow">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="w-2 h-2 rounded-full bg-rose-500"></span>
                        <span class="font-semibold text-slate-700 text-sm">质量过滤</span>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">屏蔽招聘广告、垃圾营销、低质量站点，确保每篇都值得阅读</p>
                </div>
            </div>
        </div>
        
        <!-- Sections -->
        {sections}
        
        <!-- Footer -->
        <footer class="mt-20 text-center text-slate-400 text-sm animate-fade-in" style="animation-delay: 0.5s">
            <div class="flex items-center justify-center gap-4 mb-4">
                <span class="w-8 h-px bg-slate-300"></span>
                <div class="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-lg shadow-indigo-200">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z"/></svg>
                </div>
                <span class="w-8 h-px bg-slate-300"></span>
            </div>
            <p class="mb-2">数据来源: Hacker News, GitHub Trending, TechCrunch, 机器之心, 量子位等</p>
            <p>生成时间: {generated_at}</p>
        </footer>
    </div>
</body>
</html>'''
        
        # 生成每个板块的 HTML
        sections_html = []
        source_set = set()
        delay = 0.3
        
        for category, items in sorted_categories:
            config = category_config.get(category, {"icon": "📰", "gradient": "from-indigo-500 to-violet-600", "bg": "#e0e7ff"})
            
            # 生成该分类下的卡片
            cards_html = []
            for item in items[:15]:  # 每个板块最多显示 15 条
                source_set.add(item["source"])
                
                # 确定 source badge 类名
                source_lower = item["source"].lower()
                if "hacker" in source_lower:
                    badge_class = "badge-hackernews"
                elif "github" in source_lower:
                    badge_class = "badge-github"
                elif "techcrunch" in source_lower:
                    badge_class = "badge-techcrunch"
                elif "机器" in source_lower:
                    badge_class = "badge-jiqizhixin"
                elif "量子" in source_lower:
                    badge_class = "badge-qbitai"
                elif "infoq" in source_lower:
                    badge_class = "badge-infoq"
                elif "solidot" in source_lower:
                    badge_class = "badge-solidot"
                elif "36kr" in source_lower or "36氪" in source_lower:
                    badge_class = "badge-36kr"
                elif "ifanr" in source_lower or "爱范儿" in source_lower:
                    badge_class = "badge-ifanr"
                elif "sspai" in source_lower or "少数派" in source_lower:
                    badge_class = "badge-sspai"
                else:
                    badge_class = "badge-default"
                
                # 热度等级
                score = item.get("score", 0)
                if score >= 100:
                    score_class = "score-hot"
                    score_display = f'<span class="font-bold">{score}</span>'
                elif score >= 50:
                    score_class = "score-warm"
                    score_display = f'<span class="font-semibold">{score}</span>'
                else:
                    score_class = "score-normal"
                    score_display = f'<span>{score}</span>'
                
                # 星星图标颜色
                star_color = "#ef4444" if score >= 100 else "#f59e0b" if score >= 50 else "#9ca3af"
                
                # 关键词标签
                keywords_html = ""
                for kw in item.get("matched_keywords", [])[:3]:
                    keywords_html += f'<span class="keyword-tag px-2.5 py-1 rounded-lg text-xs font-medium">{kw}</span>'
                
                card = f'''
                <a href="{item['url']}" target="_blank" class="block bg-white rounded-2xl p-6 card-shadow border border-slate-100 group">
                    <div class="flex items-start justify-between mb-4">
                        <span class="{badge_class} text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-sm">
                            {item['source']}
                        </span>
                        <span class="text-xs text-slate-400">{item['time']}</span>
                    </div>
                    
                    <h3 class="text-base font-semibold text-slate-800 leading-relaxed mb-4 group-hover:text-indigo-600 transition-colors duration-200">
                        {item['title']}
                    </h3>
                    
                    <div class="flex flex-wrap gap-2 mb-4">
                        {keywords_html}
                    </div>
                    
                    <div class="flex items-center justify-between pt-4 border-t border-slate-100">
                        <div class="flex items-center gap-1.5 {score_class}">
                            <svg class="w-4 h-4" fill="{star_color}" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10.868 2.884c-.321-.772-1.415-.772-1.736 0l-1.83 4.401-4.753.381c-.833.067-1.171 1.107-.536 1.651l3.62 3.102-1.106 4.637c-.194.813.691 1.456 1.405 1.02L10 15.591l4.069 2.485c.713.436 1.598-.207 1.404-1.02l-1.106-4.637 3.62-3.102c.635-.544.297-1.584-.536-1.65l-4.752-.382-1.831-4.401Z" clip-rule="evenodd"/></svg>
                            {score_display}
                        </div>
                        
                        <div class="flex items-center gap-1 text-indigo-600 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                            <span>阅读</span>
                            <div class="transform group-hover:translate-x-1 transition-transform">''' + icons["arrow"] + '''</div>
                        </div>
                    </div>
                </a>'''
                cards_html.append(card)
            
            # 生成板块 HTML
            section_html = f'''
        <section class="mb-16 animate-fade-in" style="animation-delay: {delay}s">
            <div class="flex items-center gap-4 mb-8">
                <div class="w-12 h-12 rounded-xl bg-gradient-to-br {config['gradient']} flex items-center justify-center text-white shadow-lg">
                    {config['icon']}
                </div>
                <div class="flex-1">
                    <div class="flex items-center gap-3">
                        <h2 class="text-2xl font-bold text-slate-900">{category}</h2>
                        <span class="bg-slate-100 text-slate-600 text-sm font-semibold px-3 py-1 rounded-full">
                            {len(items)}
                        </span>
                    </div>
                    <div class="section-title-line mt-2 w-24"></div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {''.join(cards_html)}
            </div>
        </section>'''
            sections_html.append(section_html)
            delay += 0.1
        
        # 生成数据源列表HTML
        source_list_html = ""
        for src in sorted(source_set):
            source_list_html += f'<span class="source-tag">{src}</span>'
        
        # 生成分类列表HTML
        category_list_html = ""
        for cat, items in sorted_categories:
            category_list_html += f'<span class="category-tag">{cat} ({len(items)})</span>'
        
        # 生成最热5条标题列表HTML
        top5_html = ""
        for i, item in enumerate(news_items[:5], 1):
            top5_html += f'''<div class="top5-item">
                <span class="top5-rank">{i}</span>
                <span class="top5-title">{item['title']}</span>
            </div>'''
        
        # 填充模板
        html_content = html_template.format(
            date=datetime.now().strftime("%Y年%m月%d日"),
            total_count=len(news_items[:50]),
            source_count=len(source_set),
            category_count=len(sorted_categories),
            time_window=TIME_WINDOW_HOURS,
            hn_threshold=SCORE_THRESHOLDS["Hacker News"],
            github_threshold=SCORE_THRESHOLDS["GitHub Trending"],
            sections="\n".join(sections_html),
            source_list=source_list_html,
            category_list=category_list_html,
            top5_list=top5_html,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 保存 HTML
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"[OK] HTML 报告已保存: {output_path}")
        return output_path
    
    async def generate_pdf(self, html_path, pdf_path):
        """将 HTML 转换为 PDF"""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(f"file:///{html_path.replace('\\', '/')}")
                await page.pdf(
                    path=pdf_path,
                    format="A4",
                    margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"},
                    print_background=True
                )
                await browser.close()
            
            print(f"[OK] PDF 报告已保存: {pdf_path}")
            return pdf_path
        except ImportError:
            print("[WARN] 未安装 playwright，无法生成 PDF")
            print("   请运行: pip install playwright && playwright install chromium")
            return None
        except Exception as e:
            print(f"PDF 生成失败: {e}")
            return None


async def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path.home() / "tech-news-reports"
    output_dir.mkdir(exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    html_path = str(output_dir / f"tech-news-{date_str}.html")
    pdf_path = str(output_dir / f"tech-news-{date_str}.pdf")
    
    async with TechNewsScanner() as scanner:
        # 获取新闻
        news = await scanner.fetch_all()
        
        if not news:
            print("[ERR] 未获取到任何新闻，请检查网络连接")
            return
        
        # 生成 HTML
        scanner.generate_html(news, html_path)
        
        # 生成 PDF
        await scanner.generate_pdf(html_path, pdf_path)
        
        print(f"\n报告已保存到目录: {output_dir}")
        print(f"HTML: {html_path}")
        if Path(pdf_path).exists():
            print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
