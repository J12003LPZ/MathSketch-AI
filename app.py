# app.py
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json
from PIL import Image
import tempfile
import google.generativeai as genai  # Ensure that genai is installed and properly configured
import logging  # For error logging
import re  # For regex operations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the genai client with the API key
genai.configure(api_key=API_KEY)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Function to validate allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'heic', 'heif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        logger.error("No image part in the request")
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        logger.error("No selected image")
        return jsonify({'error': 'No selected image'}), 400

    if not allowed_file(file.filename):
        logger.error("Unsupported file type")
        return jsonify({'error': 'Unsupported file type.'}), 400

    try:
        # Read image data
        image = Image.open(file.stream).convert('RGB')

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            image.save(temp_file, format='PNG')
            temp_file_path = temp_file.name

        # Upload the image to the Gemini API
        myfile = genai.upload_file(temp_file_path, mime_type='image/png')

        # Delete the temporary file after uploading
        os.unlink(temp_file_path)

        # Define variables if any (empty in this case)
        variables = {}

        # Generate the prompt
        variables_str = json.dumps(variables, ensure_ascii=False)
        prompt = (
            f"You have been given an image with various mathematical expressions, equations, or graphical problems, including geometry-related questions like the Pythagorean theorem, area, and volume calculations. "
            f"Use the PEMDAS rule for solving standard mathematical expressions: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). "
            f"For geometry problems, apply relevant theorems and formulas, such as the Pythagorean theorem (a^2 + b^2 = c^2), area formulas for different shapes (e.g., triangles, circles, rectangles), and volume formulas for solids (e.g., cubes, spheres, cylinders). "
            f"For example: "
            f"Q. 2 + 3 * 4 "
            f"A. 2 + 3 * 4 = 14 "
            f"Q. a^2 + b^2 "
            f"A. If a = 3 and b = 4, then a^2 + b^2 = 25 "
            f"Q. Area of a circle with radius r "
            f"A. If r = 5, then Area = π * r^2 = 78.54 "
            f"Return your answer in strict JSON format as a list of dictionaries, each containing both the 'question' and 'result' fields. "
            f"For example: [{{\"question\": \"2 + 3 * 4\", \"result\": 14}}, {{\"question\": \"a^2 + b^2\", \"result\": 25}}, {{\"question\": \"Area of a circle with radius r\", \"result\": 78.54}}] "
            f"Ensure that the 'question' field contains only the mathematical expression or geometry problem without any trailing symbols like '?', '=', or similar. "
            f"Analyze the equations, expressions, and geometry problems in this image and return the answers accordingly: "
            f"Here is a dictionary of user-assigned variables. If the given expression has any of these variables, use its actual value from this dictionary accordingly: {variables_str}. "
            f"DO NOT USE BACKTICKS, MARKDOWN FORMATTING, OR ANY TEXT OTHER THAN THE JSON FORMATTED ANSWER. "
            f"Ensure all keys and string values use double quotes for JSON compatibility."
        )

        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")

        # Generate content (pass a list containing the prompt and the file)
        response = model.generate_content([prompt, myfile])
        logger.info(f"Raw AI Response: {response.text}")


       # Parse the AI response to extract 'question' and 'result' values
        try:
            parsed_response = json.loads(response.text)
            if isinstance(parsed_response, list):
                # Create a list of "question = result" strings after cleaning the question
                results = [
                    f"{re.sub(r'\s*[=?]+\s*$', '', item['question'])} = {item['result']}" 
                    for item in parsed_response 
                    if 'question' in item and 'result' in item
                ]
                # Join multiple results into a newline-separated string for better readability
                results_str = '\n'.join(results)
            elif isinstance(parsed_response, dict) and 'question' in parsed_response and 'result' in parsed_response:
                # Clean the single question
                question_clean = re.sub(r'\s*[=?]+\s*$', '', parsed_response['question'])
                results_str = f"{question_clean} = {parsed_response['result']}"
            else:
                # The AI returned an unexpected format
                logger.warning(f"Unexpected AI response format: {parsed_response}")
                results_str = "Could not parse the AI response."
        except json.JSONDecodeError as parse_error:
            logger.error(f"JSON decoding failed: {parse_error}")
            logger.error(f"AI Response Text: {response.text}")
            results_str = "An error occurred while processing the AI response."

        # Return the formatted results to the frontend
        return jsonify({'status': 'Image received and being processed.', 'ai_response': results_str}), 200

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()  # Print the stack trace in the server console for debugging
        return jsonify({'error': 'Internal server error. Please check the server logs.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
