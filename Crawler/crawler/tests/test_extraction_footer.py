from crawler.config import CrawlConfig
from crawler.extraction import extract_heading_blocks


def test_extract_heading_blocks_ignores_footer_content() -> None:
    html = """
    <html>
      <body>
        <main>
          <h2>Body Section</h2>
          <p>Keep this paragraph.</p>
          <footer>
            <ul>
              <li>Drop this footer item.</li>
            </ul>
          </footer>
        </main>
      </body>
    </html>
    """

    blocks = extract_heading_blocks(html, CrawlConfig())

    assert len(blocks) == 1
    assert blocks[0]["section_path"] == ["Body Section"]
    assert "Keep this paragraph." in blocks[0]["text"]
    assert "Drop this footer item." not in blocks[0]["text"]
