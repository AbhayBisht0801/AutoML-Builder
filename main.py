from flask import Flask, render_template, request, send_file, Response, stream_with_context
import os
import io
import logging
import threading
from io import StringIO
from Agent import MLAagent  # Assuming this is your existing Agent class
import time
processing_thread = None
# Create Flask app
app = Flask(__name__)

# Configure a thread-safe log buffer
log_stream = StringIO()

# Configure standard logging
logger = logging.getLogger("my_logger")
log_handler = logging.StreamHandler(log_stream)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle file and description submission."""
    global processing_thread
    
    # Get file and description
    file = request.files['file']
    description = request.form['description']
    
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Start MLAagent in a separate thread
    def run_agent():
        agent = MLAagent(problem_statement=description, dataset_path=file_path, logger=logger)
        agent.execute_all()
    
    processing_thread = threading.Thread(target=run_agent)
    processing_thread.start()

    return render_template('logs.html', file=file)

@app.route('/stream-logs')
def stream_logs():
    def generate():
        global processing_thread
        if processing_thread is None:
            return "data: No processing thread found.\n\n"
            
        log_stream.seek(0)
        while processing_thread.is_alive() or log_stream.tell() != log_stream.seek(0, 2):
            log_stream.seek(0)
            lines = log_stream.read()
            if lines:
                formatted_lines = lines.replace('\n', '<br>')
                yield 'data: {}\n\n'.format(formatted_lines)
                log_stream.truncate(0)
                log_stream.seek(0)
            time.sleep(1)
        yield "data: [The Process is Completed.Your Project Folder is available in Project Creation]\n\n"
    
    return Response(generate(), content_type='text/event-stream')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False)