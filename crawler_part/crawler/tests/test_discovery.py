from crawler.config import CrawlConfig
from crawler.discovery import extract_framework_links
from crawler.utils import normalize_url


def test_normalize_url_strips_query_and_fragment() -> None:
    assert (
        normalize_url("/miniprogram/dev/reference/api/?a=1#intro", "https://developers.weixin.qq.com")
        == "https://developers.weixin.qq.com/miniprogram/dev/reference/api"
    )


def test_extract_framework_links_from_fixture() -> None:
    html = """
    <html>
      <body>
        <aside>
          <div class="menu-group">
            <div class="menu-title">框架</div>
            <ul>
              <li><a href="/miniprogram/dev/reference/api/App.html">App</a></li>
              <li><a href="/miniprogram/dev/reference/api/Page.html">Page</a></li>
              <li><a href="/miniprogram/dev/reference/api/Page.html#dup">Page duplicate</a></li>
            </ul>
          </div>
        </aside>
        <main>
          <a href="/miniprogram/dev/reference/api/Outside.html">Outside</a>
        </main>
        <main>
          <a href="/miniprogram/dev/reference/api/Outside.html">Outside</a>
        </main>
      </body>
    </html>
    """
    links = extract_framework_links(html, CrawlConfig())
    assert links == [
        "https://developers.weixin.qq.com/miniprogram/dev/reference/api/App.html",
        "https://developers.weixin.qq.com/miniprogram/dev/reference/api/Page.html",
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
