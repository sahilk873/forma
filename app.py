# backend/app.py
from flask import Flask, request, jsonify
import os
from interpret import run_exercise_analysis

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ensure the video file was provided
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400

    video = request.files['video']
    
    # Retrieve additional parameters
    requires_mirroring = request.form.get('requiresMirroring', 'false').lower() == 'true'
    exercise = request.form.get('exercise', '')

    # Save the uploaded video to disk (this is optional and depends on your processing needs)
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    # --- Video processing and analysis logic goes here ---
    # For demonstration, we simulate an analysis result.

    analysis_result = run_exercise_analysis(video_path)
    
    # Optionally, remove the saved file after processing:
    # os.remove(video_path)

    return jsonify({'success': True, 'data': analysis_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
