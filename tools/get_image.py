import os
import mimetypes
import google.generativeai as genai

def get_image(image_path: str):
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            return {"success": False, "error": f"Could not detect image MIME type for {image_path}"}

        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"success": False, "error": "GOOGLE_API_KEY environment variable not set."}
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content([
            {
                "mime_type": mime_type,
                "data": image_bytes
            },
            {
                "text": (
                    "Describe this image in detail. "
                    "If it has a table, extract it as Markdown. "
                    "If it has a chart or graph, note down key observations. "
                    "Return in markdown format."
                )
            }
        ])

        return {"success": True, "description": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}
