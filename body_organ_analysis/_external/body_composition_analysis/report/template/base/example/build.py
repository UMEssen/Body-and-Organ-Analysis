import tempfile
from pathlib import Path

import jinja2
import weasyprint

if __name__ == "__main__":
    # Define the Jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            [
                Path(__file__).parent,
                Path(__file__).parent.parent / "template",
            ]
        ),
        autoescape=jinja2.select_autoescape(["html", "jinja"]),
    )

    template = env.get_template("report.html.jinja")
    base_url = Path(__file__).parent.parent / "template"

    with tempfile.TemporaryDirectory() as tempdir:
        for theme in ("light", "dark"):
            html_content = template.render(
                app_version="?.?.?",
                contact_email="ship-ai@uk-essen.de",
                theme=theme,
            )
            html_file = Path(tempdir) / f"index_{theme}.html"
            html_file.write_text(html_content, "utf-8")
            weasyprint.HTML(filename=html_file, base_url=base_url).write_pdf(
                Path(__file__).parent / f"report_{theme}.pdf"
            )
