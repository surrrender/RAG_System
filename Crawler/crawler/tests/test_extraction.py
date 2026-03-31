from crawler.config import CrawlConfig
from crawler.extraction import extract_heading_blocks, extract_page_content


DOC_HTML = """
<html>
  <head><title>App Reference</title></head>
  <body>
    <nav class="breadcrumb">
      <span>文档</span><span>框架</span><span>App</span>
    </nav>
    <main>
      <h1>App</h1>
      <p>更新时间：2026-03-01</p>
      <h2>注册</h2>
      <p>用于注册小程序。</p>
      <pre><code>App({})</code></pre>
      <h3>参数</h3>
      <p>接受一个对象参数。</p>
    </main>
  </body>
</html>
"""


def test_extract_page_content() -> None:
    page = extract_page_content(DOC_HTML, "https://example.com/app", CrawlConfig())
    assert page.title == "App"
    assert page.nav_path == ["文档", "框架", "App"]
    assert page.updated_at == "2026-03-01"
    assert "用于注册小程序" in page.raw_text
    assert page.code_blocks == ["App({})"]


def test_extract_heading_blocks() -> None:
    blocks = extract_heading_blocks(DOC_HTML, CrawlConfig())
    by_path = {tuple(block["section_path"]): block for block in blocks}
    assert ("注册",) in by_path
    assert "用于注册小程序" in by_path[("注册",)]["text"]
    assert by_path[("注册",)]["code_blocks"] == ["App({})"]
    assert ("注册", "参数") in by_path


def test_extract_nav_path_from_breadcrumb_component() -> None:
    html = """
    <html>
      <head><title>Feature</title></head>
      <body>
        <div class="Breadcrumb">
          <span class="breadcrumb-item">
            <span class="breadcrumb-inner is-link">Skyline 渲染引擎</span>
            <span class="breadcrumb-separator">/</span>
          </span>
          <span class="breadcrumb-item">
            <span class="breadcrumb-inner is-link">概览</span>
            <span class="breadcrumb-separator">/</span>
          </span>
          <span class="breadcrumb-item">
            <span class="breadcrumb-inner is-link">特性</span>
            <span class="breadcrumb-separator">/</span>
          </span>
        </div>
        <main>
          <h1>Feature</h1>
          <h2>Section</h2>
          <p>content</p>
        </main>
      </body>
    </html>
    """
    page = extract_page_content(html, "https://example.com/feature", CrawlConfig())
    assert page.nav_path == ["Skyline 渲染引擎", "概览", "特性"]
