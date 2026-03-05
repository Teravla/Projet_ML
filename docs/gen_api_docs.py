import os
import sys
import mkdocs_gen_files

SRC_ROOT = os.path.join(os.path.dirname(__file__), "..", "src")
DOCS_API_ROOT = "api"

sys.path.insert(0, os.path.abspath(SRC_ROOT))


def generate_api_docs(write_to_disk: bool = True) -> None:
    """
    Generate API documentation for Python modules under SRC_ROOT.
    """
    print("Generating API documentation...")

    if write_to_disk and os.path.exists(DOCS_API_ROOT):
        for root, dirs, files in os.walk(DOCS_API_ROOT, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(DOCS_API_ROOT)

    for root, dirs, files in os.walk(SRC_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith("__")]
        if "__pycache__" in root:
            continue

        rel_dir = os.path.relpath(root, SRC_ROOT)
        doc_dir = (
            os.path.join(DOCS_API_ROOT, rel_dir) if rel_dir != "." else DOCS_API_ROOT
        )
        os.makedirs(doc_dir, exist_ok=True)

        py_files = [f for f in files if f.endswith(".py") and not f.startswith("__")]

        if not py_files:
            continue

        for py_file in py_files:
            full_path = os.path.join(root, py_file)
            module_name = os.path.relpath(full_path, SRC_ROOT)[:-3].replace(os.sep, ".")
            doc_path = os.path.join(doc_dir, py_file[:-3] + ".md")

            page_title = py_file[:-3]

            content = f"# {page_title}\n\n::: {module_name}\n"

            with mkdocs_gen_files.open(doc_path, "w") as f:
                f.write(content)
            if write_to_disk:
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(content)

            mkdocs_gen_files.set_edit_path(doc_path, full_path)

    print("API documentation generation complete.")


generate_api_docs()
