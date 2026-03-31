from crawler.config import CrawlConfig
from crawler.discovery import extract_framework_links, extract_subnavbar_links
from crawler.utils import normalize_url


def test_normalize_url_strips_query_and_fragment() -> None:
    assert (
        normalize_url("/miniprogram/dev/reference/api/?a=1#intro", "https://developers.weixin.qq.com")
        == "https://developers.weixin.qq.com/miniprogram/dev/reference/api"
    )


def test_extract_subnavbar_links_from_framework_page() -> None:
    html = """
    <html>
      <body>
        <div class="subnavbar">
          <ul class="subnavbar__list">
            <li><a href="/miniprogram/dev/framework/">指南</a></li>
            <li><a href="/miniprogram/dev/reference/">框架</a></li>
            <li><a href="/miniprogram/dev/component/">组件</a></li>
            <li><a href="/miniprogram/dev/api/">API</a></li>
            <li><a href="/miniprogram/dev/server/API/">服务端</a></li>
          </ul>
        </div>
      </body>
    </html>
    """

    links = extract_subnavbar_links(html, CrawlConfig())

    assert links == [
        "https://developers.weixin.qq.com/miniprogram/dev/framework",
        "https://developers.weixin.qq.com/miniprogram/dev/reference",
        "https://developers.weixin.qq.com/miniprogram/dev/component",
        "https://developers.weixin.qq.com/miniprogram/dev/api",
    ]


def test_extract_framework_links_filters_to_same_section() -> None:
    html = """
    <html>
      <body>
        <aside>
          <a href="/miniprogram/dev/reference/api/App.html">App</a>
          <a href="/miniprogram/dev/reference/api/Page.html#dup">Page</a>
          <a href="/miniprogram/dev/component/view.html">Component outside</a>
        </aside>
      </body>
    </html>
    """

    links = extract_framework_links(
        html,
        CrawlConfig(),
        base_url="https://developers.weixin.qq.com/miniprogram/dev/reference/",
    )

    assert links == [
        "https://developers.weixin.qq.com/miniprogram/dev/reference/api/App.html",
        "https://developers.weixin.qq.com/miniprogram/dev/reference/api/Page.html",
    ]


def test_extract_framework_links_supports_framework_section() -> None:
    html = """
    <html>
      <body>
        <aside>
          <a href="/miniprogram/dev/framework/quickstart/">快速开始</a>
          <a href="/miniprogram/dev/framework/config.html">配置</a>
          <a href="/miniprogram/dev/api/base/wx.canIUse.html">API outside</a>
        </aside>
      </body>
    </html>
    """

    links = extract_framework_links(
        html,
        CrawlConfig(),
        base_url="https://developers.weixin.qq.com/miniprogram/dev/framework/",
    )

    assert links == [
        "https://developers.weixin.qq.com/miniprogram/dev/framework/quickstart",
        "https://developers.weixin.qq.com/miniprogram/dev/framework/config.html",
    ]


def test_extract_framework_links_returns_empty_when_sidebar_missing() -> None:
    html = """
    <html>
      <body>
        <main>
          <a href="/miniprogram/dev/reference/api/App.html">App</a>
        </main>
      </body>
    </html>
    """

    assert extract_framework_links(html, CrawlConfig()) == []
