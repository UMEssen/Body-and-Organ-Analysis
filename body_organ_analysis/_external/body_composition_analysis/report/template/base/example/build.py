import tempfile
from pathlib import Path

import jinja2
import weasyprint

if __name__ == "__main__":
    # Define the Jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            [
                str(Path(__file__).parent),
                str(Path(__file__).parent.parent / "template"),
            ]
        ),
        autoescape=jinja2.select_autoescape(["html", "jinja"]),
    )

    # Load the derived report template and render HTML using Jinja2
    template = env.get_template("report.html.jinja")
    html_content = template.render(
        app_version="?.?.?",
        model_version="?.?.?",
        contact_email="ship-ai@uk-essen.de",
    )

    with tempfile.TemporaryDirectory() as tempdir:
        # Write HTML file to disk in order to be able to include relative files
        # (e.g. generated images in the same temporary directory)
        html_file = Path(tempdir) / "index.html"
        with html_file.open("w") as ofile:
            ofile.write(html_content)

        # Create Weasyprint document and render as PDF file
        document = weasyprint.HTML(
            filename=html_file,
            base_url=str(Path(__file__).parent.parent / "template"),
        )
        document.write_pdf(Path(__file__).parent / "report.pdf")
        document.write_png(str(Path(__file__).parent / "report.png"))
