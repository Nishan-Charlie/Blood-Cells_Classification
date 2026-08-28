"""Assemble the filled dissertation .docx from the chapter modules."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_common import new_doc, add_page_numbers          # noqa: E402
from ch01 import front_matter, chapter1                     # noqa: E402
from ch02 import chapter2                                   # noqa: E402
from ch03 import chapter3                                   # noqa: E402
from ch04 import chapter4                                   # noqa: E402
from ch05 import chapter5, references                       # noqa: E402
from appendices import appendices                           # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "thesis" / "Dissertation.docx"

doc = new_doc()
front_matter(doc)
chapter1(doc)
chapter2(doc)
chapter3(doc)
chapter4(doc)
chapter5(doc)
references(doc)
appendices(doc)
add_page_numbers(doc)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)

# ---- word count of body text (excludes tables, which python-docx keeps apart)
words = sum(len(p.text.split()) for p in doc.paragraphs)
print("body paragraph words:", words)
print("paragraphs:", len(doc.paragraphs), " tables:", len(doc.tables))
