"""
基于 AI 语义判断的检索标注生成脚本
根据对 40 个问题及检索 chunk 的语义分析，判断每个 chunk 是否相关。
"""
from __future__ import annotations
import json, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load raw chunk data
raw = json.loads((OUTPUT_DIR / "_raw_chunks_for_annotation.json").read_text(encoding="utf-8"))

# ============================================================
# 相关性判断（严格语义：宁可少选，不可误判）
# 每个 question_id → set of relevant chunk_ids
# ============================================================
RELEVANT = {
    "q001": {
        # 列举常用配置项：页面路径、窗口表现、网络超时、底部 tab
        "0503aa8edba13803-ab8a6820b7d58cc85d62",
        "953c22d4d25b5d58-67b7f26bc3e23803abe5",
        # style: "v2" 配置项
        "d118c2b671de544d-6133dc5d4aefe08da91d",
    },
    "q002": {
        # project.config.json 的作用：保存开发者工具个性化配置
        "0503aa8edba13803-a1c2b44b5302eba39bc6",
    },
    "q003": {
        # 页面生命周期函数：onLoad, onShow, onRouteDone, onHide, onUnload
        "565c0da2f4365e12-87ea65343fe64a79935e",
    },
    "q004": {
        # 小程序运行机制：冷启动/热启动/前台/后台/挂起/销毁
        "600ad658659816bf-ef08c1d664cd417f295f",
        "600ad658659816bf-7c0ae045141bc7f5e4e9",
        "600ad658659816bf-498fd7881e183a34e394",
        "600ad658659816bf-94c1015e7fac47ddf94f",
        "600ad658659816bf-f09fc0b589f8df6cde16",
        "600ad658659816bf-7abd9d2d6b01b5ed0f0d",
        "600ad658659816bf-2625d9a3c5eb2cd742fe",
        "600ad658659816bf-7e5767a5c3901ad5cf5",
    },
    "q005": set(),  # 无相关 chunk：检索结果未命中"不能操作 DOM"的原因解释
    "q006": set(),  # 无相关 chunk：检索结果未比较 WXML 与 HTML 区别
    "q007": {
        # WXSS 具有 CSS 大部分特性并做了扩充修改（rpx单位等）
        "b8dc6d86519bbf84-630e920",
        "0503aa8edba13803-b56d06e",
    },
    "q008": {
        # 数据绑定语法 Mustache {{}}
        "e803f0c65a7facb6-6f678b1",
        "45d2b2cd078105cb-5c6456e",
        "e803f0c65a7facb6-800fa18",
        "e803f0c65a7facb6-8de65d7",
        "e803f0c65a7facb6-29b9d5c",
        "e803f0c65a7facb6-bb6216d",
        "e803f0c65a7facb6-6e641ee",
        "e803f0c65a7facb6-5ac21b4",
        "45d2b2cd078105cb-fbebfa7",
    },
    "q009": {
        # 事件机制：视图层到逻辑层的通讯方式
        "c703f3af8007f07a-03fec4f",
    },
    "q010": {
        # 页面栈：组织为页面栈+悬垂页面的组合形式
        "565c0da2f4365e12-5920b1a",
    },
    "q011": {
        # wx:if 条件切换时销毁或重新渲染，是惰性的；hidden 始终渲染
        "969ffa36dc67f1fa-055ef1c",
    },
    "q012": {
        # 列表渲染：wx:key 保持项目特征和状态
        "757ddcb56f4e65ca-64fad3a",
    },
    "q013": {
        # block 标签包装多个组件，用于 wx:if/wx:for
        "969ffa36dc67f1fa-5a3e485",
    },
    "q014": {
        # template 模板：定义代码片段，用 is 和 data 调用
        "bdb07440571a0dd8-eb3844e",
        "bdb07440571a0dd8-a673b98",
        "bdb07440571a0dd8-ab2a654",
        "31bc6fedcffcefa3-3b4dcaf",
    },
    "q015": {
        # slot 插槽机制：承载使用者提供的 wxml 结构，可多 slot
        "d21d59199f0eaf0d-307a492",
        "178ade0c1ee9634a-307d5a4",
    },
    "q016": {
        # 自定义组件：注册、引用声明、组件化编程
        "1e40635cc3e4a712-5763ff0",
        "1e40635cc3e4a712-0d236b3",
    },
    "q017": {
        # properties, data, methods 的作用
        # 没有任何 chunk 直接解释这组概念
    },
    "q018": set(),  # 无相关 chunk：未直接说明 properties 和 data 的区别
    "q019": set(),  # 无相关 chunk：组件生命周期在检索结果中未充分体现
    "q020": set(),  # 无相关 chunk：component 和 page 生命周期区别未找到
    "q021": {
        # 页面间通信：数据通道，getOpenerEventChannel
        "5f903b55b5c4d15e-e86d559",
        "5f903b55b5c4d15e-6259a29",
    },
    "q022": set(),  # behaviors 概念未在检索结果中充分体现
    "q023": {
        # 页面跳转方式
        "565c0da2f4365e12-3a2a061",
        "5d9e47199c7914dc-7a8324c",
    },
    "q024": {
        # reLaunch 和 navigateBack 的作用
        "5d9e47199c7914dc-7a8324c",
    },
    "q025": {
        # 页面传参方式：页面间通信、数据通道
        "5f903b55b5c4d15e-e86d559",
        "5f903b55b5c4d15e-6259a29",
    },
    "q026": {
        # 不能跳到 tabbar 页面
        "5d9e47199c7914dc-7a8324c",
    },
    "q027": {
        # 页面间通信：数据通道，getOpenerEventChannel
        "5f903b55b5c4d15e-e86d559",
        "5f903b55b5c4d15e-6259a29",
    },
    "q028": {
        # 域名要求：只支持 https/wss，不能使用 IP
        "fc7d0a472bb34b83-a1b811a",
        "fc7d0a472bb34b83-2d18920",
    },
    "q029": {
        # request, uploadFile, downloadFile 各自功能
        "63cd56e045d3f346-9c4cddb",
        "b9d59b435ca01f63-81f1adc",
        "b9d59b435ca01f63-143c494",
        "4627cafd6727e311-4421edb",
    },
    "q030": {
        # 获取 code：wx.login 获取临时登录凭证 code；UnionID 获取
        "4cee1d71bcbbfdbf-e3d2205",
    },
    "q031": {
        # wx.login 和 wx.getUserProfile 的关系
        "79e4b3f05f3f22a4-8a17a94",
        "79e4b3f05f3f22a4-c3b9064",
        "deeec4007de6dd4d-32d7949",
        "e52c1d6af90c0e54-a3fcd2e",
    },
    "q032": {
        # 小程序登录流程：wx.login → code → code2Session → OpenID/UnionID
        "8326435fdf06e36d-4a3964f",
        "8326435fdf06e36d-2171e17",
        "8326435fdf06e36d-20d2250",
        "8326435fdf06e36d-5e516ad",
    },
    "q033": {
        # 如何实现 token 登录
        "7af8dc522f644744-21fa14a",
        "8326435fdf06e36d-5e516ad",
    },
    "q034": {
        # 如何使用本地缓存 API：setStorage/getStorage/clearStorage等
        "66ea76ae2e854d4d-e774702",
        "f132c20ba9c25395-9cc2aba",
        "fb676d10b1fe6544-84eb4e3",
        "ef6e20af66bf9a33-94c4f55",
        "72175907e04730c9-1bbd98a",
        "f9db77d01bc73b1c-eae87be",
        "63b5b5a86c2da7fd-4c00f2d",
    },
    "q035": {
        # 同步缓存和异步缓存区别：Sync 后缀为同步版本
        "72175907e04730c9-1bbd98a",
        "126bff8bbd7ccd45-1f9815f",
        "fb676d10b1fe6544-84eb4e3",
        "4bc371f7e98d7fee-6b3d8a0",
        "9444b873255d94e4-6423376",
    },
    "q036": {
        # 如何上传和下载文件：wx.uploadFile / wx.downloadFile
        "63cd56e045d3f346-9c4cddb",
        "57481150b78aa92f-6a27f24",
        "63cd56e045d3f346-8d5ecf3",
        "b9d59b435ca01f63-81f1adc",
        "b9d59b435ca01f63-143c494",
        "b9d59b435ca01f63-800c607",
    },
    "q037": {
        # 图片懒加载与预览
        "5fc55133c93daeb4-8f09762",
        "ce45045fbbe50738-a8859b8",
    },
    "q038": {
        # 如何使用地图与定位 API
        "8b2d1c66eed18803-0cbb267",
        "8b2d1c66eed18803-c9f2c06",
        "673b4aa196599cbc-3a6aa4c",
        "1a5e280f35d568d5-9bf7c9c",
        "1a5e280f35d568d5-0e72240",
    },
    "q039": {
        # 调用扫码、相机、蓝牙等能力
        "344d6fa1f05c1fe1-bd84f0e",
        "a799b86da70842df-222beb8",
        "a799b86da70842df-8758cf1",
        "a799b86da70842df-8e8fafa",
        "a799b86da70842df-e25c23d",
        "a799b86da70842df-87a218e",
        "6c6a6cf2c06b628a-3c2c7c3",
    },
    "q040": {
        # 小程序性能优化
        "2443bf9b9fa1b9d7-dba85a7",
        "2443bf9b9fa1b9d7-64fd8aa",
        "843848db71855103-cc394c5",
        "843848db71855103-b9a3a68",
        "e61582a8b4dd572e-2a29175",
        "25ed487639f36dc8-d468d47",
        "9b6b8605ec5126ff-175a249",
        "e61582a8b4dd572e-96647f7",
        "c0fadf038e9fd4f8-962c743",
        "eaf34c0ea6a4af07-65f7a0a",
        "e7dd7fdcdc2acd04-97db166",
        "0a9022ab92228b25-c372ba0",
    },
}


def build_annotation(data_cfg: list[dict], reranker_enabled: bool) -> dict:
    annotations = []
    for q in data_cfg:
        qid = q["question_id"]
        relevant = RELEVANT.get(qid, set())

        retrieved = []
        relevant_ids_in_retrieved = []
        for c in q["retrieved_chunks"]:
            info = {
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "title": c["title"],
                "section_path": c["section_path"],
                "rank": c["rank"],
            }
            retrieved.append(info)
            if c["chunk_id"] in relevant:
                relevant_ids_in_retrieved.append(c["chunk_id"])

        annotations.append({
            "question_id": qid,
            "question": q["question"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "relevant_chunk_ids": relevant_ids_in_retrieved,
            "all_retrieved_chunks": retrieved,
        })

    return {
        "config": {
            "reranker_enabled": reranker_enabled,
            "top_k": 10,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "annotation_method": "ai_semantic_judgment_v3",
            "criteria": "strict semantic relevance: 宁可少选不可误判，仅当chunk文本直接帮助回答问题才标记相关",
        },
        "annotations": annotations,
    }


# Generate both configs
for cfg_key, label, filename in [
    ("no_reranker", False, "annotation_results_no_reranker_optimize.json"),
    ("with_reranker", True, "annotation_results_with_reranker_optimize.json"),
]:
    result = build_annotation(raw[cfg_key], label)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_relevant = sum(len(a["relevant_chunk_ids"]) for a in result["annotations"])
    q_with_relevant = sum(1 for a in result["annotations"] if a["relevant_chunk_ids"])
    print(f"{filename}:")
    print(f"  40 题中 {q_with_relevant} 题有相关 chunk，共标记 {total_relevant} 个相关 chunk")
    print(f"  已保存到 {path}")
