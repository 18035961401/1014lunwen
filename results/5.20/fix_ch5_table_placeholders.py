from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parent
MD = ROOT / "第五章_成都市武侯区地铁物流配送转运点布局优化研究_成本能力参数补强最终版.md"
DOCX = ROOT / "第五章_成都市武侯区地铁物流配送转运点布局优化研究_成本能力参数补强最终版.docx"


repls = {
    "表 5-X 汇总了公路直送、地铁干线、入网短驳、出网接驳、换乘作业、进出站作业、固定启用成本、班次组织成本以及碳外部成本的取值口径。": "下表汇总了公路直送、地铁干线、入网短驳、出网接驳、换乘作业、进出站作业、固定启用成本、班次组织成本以及碳外部成本的取值口径。",
    "表 5-Y 进一步给出线路级列车装载能力参数。": "下一表进一步给出线路级列车装载能力参数。",
    "上述两张参数表与表 5-5 中候选站点容量和固定启用成本共同构成公式（4-1）综合成本函数 $Z_1$ 的实证参数输入。": "上述两张参数表与候选站点属性表中的站点容量和固定启用成本共同构成公式（4-1）综合成本函数 $Z_1$ 的实证参数输入。",
}


def main():
    text = MD.read_text(encoding="utf-8")
    for old, new in repls.items():
        text = text.replace(old, new)
    MD.write_text(text, encoding="utf-8")

    doc = Document(DOCX)
    for p in doc.paragraphs:
        for old, new in repls.items():
            if old in p.text:
                # Preserve simple paragraph content; runs are not critical for these generated files.
                p.clear()
                p.add_run(p.text.replace(old, new))
    # The loop above cannot use p.text after clear, so run a second robust paragraph replacement.
    doc = Document(DOCX)
    for p in doc.paragraphs:
        original = p.text
        updated = original
        for old, new in repls.items():
            updated = updated.replace(old, new)
        if updated != original:
            p.clear()
            p.add_run(updated)
    doc.save(DOCX)
    print(MD)
    print(DOCX)


if __name__ == "__main__":
    main()
