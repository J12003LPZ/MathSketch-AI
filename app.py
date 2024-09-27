# app.py
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json
from PIL import Image
import tempfile
import google.generativeai as genai  # Ensure that genai is installed and properly configured
import logging  # For error logging

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
            f"You have been given an image with some mathematical expressions, equations, or graphical problems, and you need to solve them. "
            f"Use the PEMDAS rule for solving mathematical expressions. PEMDAS stands for the Priority Order: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). "
            f"For example: "
            f"Q. 2 + 3 * 4 "
            f"A. 14 "
            f"Q. 2 + 3 + 5 * 4 - 8 / 2 "
            f"A. 21 "
            f"Return your answer in strict JSON format as a list of dictionaries, each containing only the 'result' field. "
            f"For example: [{'"result": 14'}, {'"result": 21'}] "
            f"Analyze the equations or expressions in this image and return the answer accordingly: "
            f"Here is a dictionary of user-assigned variables. If the given expression has any of these variables, use its actual value from this dictionary accordingly: {variables_str}. "
            f"DO NOT USE BACKTICKS, MARKDOWN FORMATTING, OR ANY TEXT OTHER THAN THE JSON FORMATTED ANSWER. "
            f"Ensure all keys and string values use double quotes for JSON compatibility."
        )

        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")

        # Generate content (pass a list containing the prompt and the file)
        response = model.generate_content([prompt, myfile])

        # Log the AI response in the console
        logger.info(f"AI Response: {response.text}")

        # Parse the AI response to extract 'result' values
        try:
            parsed_response = json.loads(response.text)
            if isinstance(parsed_response, list):
                results = [item['result'] for item in parsed_response if 'result' in item]
                # Join multiple results into a comma-separated string
                results_str = ', '.join(map(str, results))
            elif isinstance(parsed_response, dict) and 'result' in parsed_response:
                results_str = str(parsed_response['result'])
            else:
                # The AI returned an unexpected format
                logger.warning(f"Unexpected AI response format: {parsed_response}")
                results_str = "Could not parse the AI response."
        except json.JSONDecodeError as parse_error:
            logger.error(f"JSON decoding failed: {parse_error}")
            logger.error(f"AI Response Text: {response.text}")
            results_str = "An error occurred while processing the AI response."

        # Return only the results to the frontend
        return jsonify({'status': 'Image received and being processed.', 'ai_response': results_str}), 200

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()  # Print the stack trace in the server console for debugging
        return jsonify({'error': 'Internal server error. Please check the server logs.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
