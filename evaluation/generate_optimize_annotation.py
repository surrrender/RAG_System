"""
基于 LLM 语义判断的检索标注生成脚本（优化版）
对 40 个问题的检索 chunk 进行严格语义相关性判断。
选择标准：仅当 chunk 文本在语义上直接帮助回答问题（即包含问题所需的实质性信息）时才标记为相关，
不因 chunk 含有问题中的关键字就认为相关。

用法: source .venv/bin/activate && python evaluation/generate_optimize_annotation.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RAW_FILE = EVAL_DIR / "outputs" / "_raw_chunks_for_annotation.json"
OUTPUT_DIR = EVAL_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw = json.loads(RAW_FILE.read_text(encoding="utf-8"))

# ============================================================================
# 相关性判断（严格语义标准）
# 仅当 chunk 的文本实质性地回答了问题或提供了问题所需的直接信息时，才标记相关
# ============================================================================
# Base relevance: chunk_ids that are always relevant regardless of mode
RELEVANT: dict[str, set[str]] = {
    # q001: app.json 常用配置项 — 需要 chunk 直接列举或描述 app.json 的配置字段
    "q001": {
        "d118c2b671de544d-9ef8bd16199aec594e30",  # 全局配置：app.json 有以下属性
        "0503aa8edba13803-ab8a6820b7d58cc85d62",  # app.json 是全局配置，包含 pages/window/tabBar 等
        "953c22d4d25b5d58-67b7f26bc3e23803abe5",  # app.json 决定页面路径、窗口表现、tab等
        "d118c2b671de544d-6133dc5d4aefe08da91d",  # app.json 中配置 "style": "v2"
        "d118c2b671de544d-62892e0b19da73d40b19",  # 指明 sitemap.json 的位置
    },

    # q002: project.config.json 的作用
    "q002": {
        "0503aa8edba13803-a1c2b44b5302eba39bc6",  # 解释了 project.config.json 的作用：保存个性化配置
    },

    # q003: 页面生命周期和应用生命周期
    "q003": {
        "0ccb9a7be5cab4a6-de565b170b09f78a54e9",  # 注册页面：指定生命周期回调
        "565c0da2f4365e12-87ea65343fe64a79935e",  # 列出页面生命周期函数：onLoad/onShow/onRouteDone/onHide/onUnload
        "600ad658659816bf-91e684f560c187a09361",  # 小程序运行机制：不同状态（应用生命周期涉及的多种状态）
    },

    # q004: 小程序的运行机制 — 所有 chunk 都直接描述运行机制
    "q004": {
        "600ad658659816bf-ef08c1d664cd417f295f",
        "600ad658659816bf-7c0ae045141bc7f5e4e9",
        "600ad658659816bf-498fd7881e183a34e394",
        "600ad658659816bf-94c1015e7fac47ddf94f",
        "600ad658659816bf-f09fc0b589f8df6cde16",
        "600ad658659816bf-7abd9d201b487bc98b7a",
        "600ad658659816bf-2625d9adc2f9cf5f2a48",
        "600ad658659816bf-7e5767aba92abe233b61",
    },

    # q005: 不能操作 DOM 的原因 — 检索结果未命中
    "q005": set(),

    # q006: WXML 和 HTML 区别 — 无 chunk 直接对比
    "q006": set(),

    # q007: WXSS 与 CSS 的区别
    "q007": {
        "b8dc6d86519bbf84-630e920ef3eff890b38f",  # WXSS 具有 CSS 大部分特性，扩充了尺寸单位和样式导入
        "0503aa8edba13803-b56d06ec2a6361cc821c",  # WXSS 扩充和修改，新增 rpx 单位
    },

    # q008: WXML 数据绑定语法 — 所有 chunk 都是关于数据绑定的示例
    "q008": {
        "e803f0c65a7facb6-6f678b1785e85087c6f9",
        "45d2b2cd078105cb-5c6456eee79e86ca9305",
        "e803f0c65a7facb6-800fa187f1e6bf4536cc",
        "e803f0c65a7facb6-8de65d711418f77df3b1",
        "e803f0c65a7facb6-29b9d5c988cdcf35b325",
        "e803f0c65a7facb6-bb6216d7c233111b7a8b",
        "e803f0c65a7facb6-6e641eee15178fa39c49",
        "e803f0c65a7facb6-5ac21b47bc3069a807b2",
        "45d2b2cd078105cb-fbebfa7d87dd2bcf3788",
    },

    # q009: 事件机制
    "q009": {
        "c703f3af8007f07a-03fec4fee2a78a0e0768",  # 事件是视图层到逻辑层的通讯方式
    },

    # q010: 页面栈
    "q010": {
        "565c0da2f4365e12-5920b1abe3819e1f4a6e",  # 小程序的页面会被组织为一个页面栈
    },

    # q011: wx:if 和 hidden 的区别
    "q011": {
        "969ffa36dc67f1fa-055ef1c82407bf089060",  # 直接对比 wx:if（惰性/切换消耗高）和 hidden（始终渲染/初始消耗高）
    },

    # q012: wx:for 和 key 的作用
    "q012": {
        "757ddcb56f4e65ca-64fad3a6c4a944e2e9ea",  # 解释 wx:for 搭配 wx:key 保持列表项状态并提高渲染效率
    },

    # q013: block 标签的作用
    "q013": {
        "969ffa36dc67f1fa-5a3e485032bf374ecd81",  # block 是包装元素，不渲染，只接受控制属性
        "757ddcb56f4e65ca-053dfc44fc7c996e9f11",  # 将 wx:for 用在 block 上渲染多节点结构块
    },

    # q014: template 模板的使用
    "q014": {
        "bdb07440571a0dd8-eb3844e03fa21538944f",  # 模板中可以定义代码片段，在不同地方调用
        "bdb07440571a0dd8-a673b98cbfd15c7ea5b5",  # 使用 is 属性声明需要的模板，传入 data
        "bdb07440571a0dd8-ab2a654e2354a7d2744c",  # 使用 name 属性作为模板名，定义代码片段
        "2eae9ab9ad4a95e6-c59076a21f60292a77d1",  # import 引用目标文件的 template
        "31bc6fedcffcefa3-3b4dcafcdd54b87ae778",  # template 定义和使用示例
    },

    # q015: slot 插槽机制
    "q015": {
        "d21d59199f0eaf0d-307a49209ae1970c9b27",  # slot 节点承载使用者提供的 wxml 结构，可多 slot
        "178ade0c1ee9634a-307d5a432ff60652662f",  # 单一 slot 和多 slot，动态 slot
    },

    # q016: 如何使用自定义组件
    "q016": {
        "1e40635cc3e4a712-5763ff03d357266deb7c",  # 支持简洁的组件化编程
        "1e40635cc3e4a712-0d236b304e885808c798",  # 在页面的 json 中使用 usingComponents 声明引用
        "1e40635cc3e4a712-8877f0aae80a168ffd76",  # 自定义组件由 json/wxml/wxss/js 组成，component: true
    },

    # q017: properties、data、methods 的作用 — 无 chunk 直接解释这三者的概念与作用
    "q017": set(),

    # q018: properties 和 data 的区别 — 无 chunk 直接对比
    "q018": set(),

    # q019: 组件生命周期
    "q019": {
        "99bf3e340f2ac3a9-18b03ddd7a4096c3f3af",  # 组件生命周期在 lifetimes 字段声明
        "99bf3e340f2ac3a9-1877557f4f5af7312fe9",  # created/attached/detached 生命周期及触发时机
        "99bf3e340f2ac3a9-a1bd21bce38e195e6847",  # 组件所在页面的生命周期：show/hide/resize/routeDone
        "99bf3e340f2ac3a9-a70f5847f29a3e649421",  # lifetimes 中 attached/detached 的示例
    },

    # q020: component 和 page 生命周期区别 — 无 chunk 直接对比两者的区别
    "q020": set(),

    # q021: 如何实现组件间通信
    "q021": {
        "0dda0c639ce9ab2f-83368ab943e76c330963",  # 三种通信方式：WXML数据绑定、事件、selectComponent
        "0dda0c639ce9ab2f-2be90ba7e468f5e22663",  # 事件系统是组件间通信的主要方式之一
        "0dda0c639ce9ab2f-a13cc4d0d90ad517c3fa",  # 父组件调用 selectComponent 获取子组件实例
        "0dda0c639ce9ab2f-c64d5a287b310adf7d5b",  # 使用 triggerEvent 触发自定义事件
    },

    # q022: behaviors 是什么
    "q022": {
        "8b4633b2b3fa91fb-3e947873a25d2993499e",  # behaviors 是组件间代码共享特性，类似 mixins
        "0a4ba510cb942edb-b109caffc87306bd1f4b",  # 注册一个 behavior
        "8b4633b2b3fa91fb-43e62f7afe0afcaec518",  # 引用内置 behavior 获得内置组件行为
        "8b4633b2b3fa91fb-44f03556f6fd9fbd3f8d",  # 同名属性/方法的覆盖规则
        "0a4ba510cb942edb-5baf4a0194e4172299ce",  # Behavior 定义示例
        "0a4ba510cb942edb-7d235821e98b0f3b338f",  # Behavior 参数定义段说明
        "0a4ba510cb942edb-4936b1d4d5e427c3d45a",  # Behavior 定义段说明
    },

    # q023: 页面跳转方式有哪些 — 无 chunk 完整列出五种跳转方式
    "q023": set(),

    # q024: reLaunch 和 navigateBack 的作用
    "q024": {
        "565c0da2f4365e12-9846cd23060457e827c7",  # navigateBack 弹出并销毁栈顶页面
        "565c0da2f4365e12-e1dcff09cffeba29bcef",  # reLaunch 销毁当前所有页面并载入新页面
        "2806cd66b0d99d93-7b46693e3bcb82aa6292",  # navigateBack 示例：delta 控制返回层级
        "2806cd66b0d99d93-ad4def30e70e0db183d3",  # navigateBack delta 示例
        "c3ac6b043446bda3-baa554c0cf3e58ec010a",  # reLaunch 示例
    },

    # q025: 页面传参方式
    "q025": {
        "5f903b55b5c4d15e-fa8349ac84e37aa5e345",  # onLoad 获取打开当前页面路径中的参数
        "5f903b55b5c4d15e-e86d559ddb1d774dfaba",  # 页面间建立数据通道 EventChannel
        "5f903b55b5c4d15e-6259a2932474dd8a3eb4",  # EventChannel 用于页面间通信传递数据
    },

    # q026: TabBar 为什么不能 navigateTo
    "q026": {
        "5d9e47199c7914dc-7a8324c3a1ad5bcfb791",  # 不能跳到 tabbar 页面
        "5d9e47199c7914dc-e5ebecaa32ab7e4ac590",  # 不能跳到 tabbar 页面
        "565c0da2f4365e12-3a2a0617e7034b004ef2",  # navigateTo 的目标必须为非 tabBar 页面
    },

    # q027: 页面之间如何通信
    "q027": {
        "5f903b55b5c4d15e-e86d559ddb1d774dfaba",  # EventChannel 建立数据通道
        "5f903b55b5c4d15e-6259a2932474dd8a3eb4",  # EventChannel emit/on 收发事件
    },

    # q028: wx.request 限制和域名要求
    "q028": {
        "fc7d0a472bb34b83-a1b811a3867cebd29260",  # 域名只支持 https/wss，不能使用 IP 地址
        "fc7d0a472bb34b83-2d18920c65d2b6d8c161",  # 并发限制 10 个
    },

    # q029: request、uploadFile、downloadFile 区别
    "q029": {
        "63cd56e045d3f346-9c4cddb4f7ced2b8b70b",  # uploadFile：将本地资源上传到服务器
        "b9d59b435ca01f63-81f1adcb990942cb2273",  # downloadFile：下载文件资源到本地
        "63cd56e045d3f346-373f63126228b9379abe",  # uploadFile 示例
        "b9d59b435ca01f63-143c494c62cd562738eb",  # downloadFile 示例
        "63cd56e045d3f346-8927d05392a9fc79a13d",  # uploadFile 监听上传进度
    },

    # q030: 如何获取 code — 无 chunk 直接展示 wx.login 获取 code
    "q030": set(),

    # q031: wx.login 和 wx.getUserProfile 的关系
    "q031": {
        "79e4b3f05f3f22a4-8a17a944de91b6d01f8e",  # getUserProfile：获取用户信息，替换 getUserInfo
        "e52c1d6af90c0e54-a3fcd2ea36b86ba2f1ca",  # wx.login 示例：获取 code 用于登录
    },

    # q032: 小程序登录流程
    "q032": {
        "8326435fdf06e36d-2171e17b6856dc384d6b",  # 小程序可通过微信登录能力获取用户身份标识
        "8326435fdf06e36d-20d225076e7c9c71bfcf",  # session_key 和 code 说明
        "8326435fdf06e36d-4a3964f69c233f4082ca",  # wx.login()→code→auth.code2Session→OpenID 完整流程
    },

    # q033: 如何实现 token 登录
    "q033": {
        "7af8dc522f644744-21fa14a29adf46d0bbef",  # wx.setBackgroundFetchToken 设置自定义 TOKEN
        "8326435fdf06e36d-5e516ad79e0f0d1eab32",  # 开发者服务器根据用户标识生成自定义登录态
    },

    # q034: 如何使用本地缓存 API
    "q034": {
        "ef6e20af66bf9a33-94c4f55994619cbe77ca",  # clearStorage 清理本地数据缓存
        "fb676d10b1fe6544-84eb4e3ae6dea0a754af",  # getStorage 异步获取
        "72175907e04730c9-1bbd98ad2f644fdbf7ce",  # getStorageSync 同步获取
        "9444b873255d94e4-4f76037ad6284bd1aa33",  # getStorageInfoSync
        "f9db77d01bc73b1c-eae87bef34f3c7feb9f4",  # batchGetStorage 批量获取
        "66ea76ae2e854d4d-e774702e1302b3387f14",  # 存储概述：setStorage/getStorage/clearStorage 等
        "2ecf0eba0c8b0dcd-a8dd573ed2f8cc6ab179",  # setStorageSync 存储数据
    },

    # q035: 同步缓存和异步缓存区别
    "q035": {
        "4bc371f7e98d7fee-6b3d8a0a6767ecb3b32d",  # clearStorageSync 是 clearStorage 的同步版本
        "126bff8bbd7ccd45-1f9815f812870dfc440c",  # getStorageInfo 异步获取
        "9444b873255d94e4-64233768ead3e2026f81",  # getStorageInfoSync 是同步版本
        "f9db77d01bc73b1c-eae87bef34f3c7feb9f4",  # batchGetStorage 异步批量获取
        "e61582a8b4dd572e-30c1682a6e833ddee64b",  # Sync 结尾为同步版本，同步 API 会阻塞 JS 线程
        "72175907e04730c9-1bbd98ad2f644fdbf7ce",  # getStorageSync 同步获取
        "fb676d10b1fe6544-84eb4e3ae6dea0a754af",  # getStorage 异步获取
    },

    # q036: 如何上传和下载文件
    "q036": {
        "57481150b78aa92f-6a27f24b7133d356ef23",  # uploadFile 示例
        "63cd56e045d3f346-9c4cddb4f7ced2b8b70b",  # uploadFile：将本地资源上传到服务器
        "63cd56e045d3f346-8d5ecf3e2aee0d80de86",  # uploadFile 监听上传进度
    },

    # q037: 图片懒加载与预览
    "q037": {
        "ce45045fbbe50738-a8859b8b6c6d19beee88",  # 屏幕外的图片使用懒加载
    },

    # q038: 地图与定位 API
    "q038": {
        "8b2d1c66eed18803-0cbb267301cc4375aca7",  # 地图服务API：地点搜索、地址解析等
        "1a5e280f35d568d5-f3f48af0ab093f601e0a",  # MapContext 获取经纬度、设置定位点
        "673b4aa196599cbc-3a6aa4c96cc4e39aad47",  # chooseLocation 打开地图选择位置
        "1a5e280f35d568d5-9bf7c9c95450033fbf86",  # MapContext 实例及操作
        "1a5e280f35d568d5-0e72240d08c3f58bcb5f",  # map 组件示例代码
        "8b2d1c66eed18803-c9f2c066e4a7f02e8969",  # map 组件功能和个性化样式
    },

    # q039: 扫码、相机、蓝牙等能力
    "q039": {
        "a799b86da70842df-8e8fafaabc64ab931919",  # 蓝牙：扫描外围设备
        "344d6fa1f05c1fe1-bd84f0e4bd5a354db23c",  # 蓝牙能力概述
        "a799b86da70842df-8758cf1f6463dded0ff8",  # 蓝牙：中心设备扫描外围设备
        "a799b86da70842df-e25c23d2d9acdb93cfe1",  # 蓝牙：主机和从机模式
        "344d6fa1f05c1fe1-4f3e7ce25bdbc66e4e42",  # 蓝牙：deviceId 说明
        "a799b86da70842df-222beb88ee953bde0dbf",  # 蓝牙：openBluetoothAdapter 初始化
        "a799b86da70842df-87a218eb8b3eaf870ddc",  # 蓝牙：getBLEDeviceServices
        "6c6a6cf2c06b628a-3c2c7c321e390e71cd8b",  # 相机
    },

    # q040: 小程序性能优化
    "q040": {
        "2443bf9b9fa1b9d7-64fd8aa580975ee86f0c",  # 性能与体验概述：启动性能和运行时性能
        "2443bf9b9fa1b9d7-dba85a72e90b0c9fd0cd",  # 如何进行性能优化
        "e61582a8b4dd572e-2a291759fed321e1c294",  # 代码注入优化
        "843848db71855103-cc394c5619fc41022b7b",  # 运行时性能：setData/渲染/页面切换/资源加载/内存优化
        "c0fadf038e9fd4f8-962c743c4a3181312073",  # 性能相关信息 API
        "25ed487639f36dc8-d468d476bf9c7cd0427e",  # 启动性能：代码包体积/代码注入/首屏渲染优化
        "9b6b8605ec5126ff-175a249b486c394b9c8c",  # 页面切换优化
        "843848db71855103-b9a3a68fcb5dc9ab731e",  # 运行时性能：各项优化方面
        "0a9022ab92228b25-c372ba024400ff7dfd95",  # 其他启动性能优化建议
        "e61582a8b4dd572e-96647f73d4cf41a84b9e",  # 避免执行复杂运算逻辑
    },
}


# Mode-specific overrides for questions where retrieved chunks differ
# (keyed by question_id, contains extra relevant chunk_ids only in this mode)
RELEVANT_EXTRA_WITH_RERANKER: dict[str, set[str]] = {
    "q034": {
        "f132c20ba9c25395-9cc2aba6",  # setStorage：将数据存储在本地缓存中
        "63b5b5a86c2da7fd-4c00f2d0",  # batchGetStorageSync：同步批量获取
    },
    "q037": {
        "5fc55133c93daeb4-8f097624",  # wx.previewImage：全屏预览图片
    },
}

# Mode-specific removals: chunk_ids that are relevant but not present in with_reranker top-10
RELEVANT_REMOVE_WITH_RERANKER: dict[str, set[str]] = {
    "q025": {
        "5f903b55b5c4d15e-fa8349ac84e37aa5e345",  # pushed out of top-10 by reranker
        "5f903b55b5c4d15e-6259a2932474dd8a3eb4",  # pushed out of top-10 by reranker
    },
    "q026": {
        "5d9e47199c7914dc-320a45955b89e712e630",  # pushed out of top-10 by reranker
    },
    "q034": {
        "9444b873255d94e4-4f76037ad6284bd1aa33",  # pushed out of top-10 by reranker
        "2ecf0eba0c8b0dcd-a8dd573ed2f8cc6ab179",  # pushed out of top-10 by reranker
    },
    "q035": {
        "fb676d10b1fe6544-84eb4e3ae6dea0a754af",  # pushed out of top-10 by reranker
    },
}


def build_annotation(data_cfg: list[dict], reranker_enabled: bool) -> dict:
    annotations = []
    for q in data_cfg:
        qid = q["question_id"]
        relevant = set(RELEVANT.get(qid, set()))
        if reranker_enabled:
            relevant |= RELEVANT_EXTRA_WITH_RERANKER.get(qid, set())
            relevant -= RELEVANT_REMOVE_WITH_RERANKER.get(qid, set())

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
            "annotation_method": "llm_semantic_judgment_optimized",
            "criteria": "strict semantic relevance: chunk must substantively help answer the question",
        },
        "annotations": annotations,
    }


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
