from bs4 import BeautifulSoup, Comment
from pathlib import Path

def get_dom_structure(html_file_path: str) -> str:
    """Extracts and formats DOM structure from the given HTML file."""
    with open(html_file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, comments
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Get simplified DOM tree text with limited depth
    def tree_text(el, depth=0, max_depth=5):
        if depth > max_depth:
            return ""
        text = "  " * depth + f"{el.name}"
        if el.attrs:
            attrs_str = " ".join(f'{k}="{v}"' for k,v in el.attrs.items())
            text += f" [{attrs_str}]"
        text += "\n"
        for child in el.children:
            if getattr(child, "name", None):
                text += tree_text(child, depth+1, max_depth)
        return text

    dom_tree_str = tree_text(soup.body or soup, 0, max_depth=5)
    return dom_tree_str

# def main():
#     file_path = "economy_of_india.html"
#     output_file = "dom.txt"
#     
#     dom_structure = get_dom_structure(file_path, max_depth=5)
#     
#     Path(output_file).write_text(dom_structure, encoding="utf-8")
#     print(f"✅ DOM structure saved to {output_file}")
# 
# if __name__ == "__main__":
#     main()
