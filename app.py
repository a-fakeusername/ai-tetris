# Import necessary classes from Flask
from flask import Flask, render_template

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the homepage ('/')
@app.route('/')
def home():
    """
    This function handles requests to the root URL '/'
    and renders the index.html template.
    """
    # The text we want to display in the textbox
    message = "Hello World"
    
    # Render the HTML template, passing the message variable to it
    # Flask automatically looks for templates in a folder named 'templates'
    return render_template('index.html', display_text=message)

# Check if the script is executed directly (not imported)
if __name__ == '__main__':
    # Run the Flask development server
    # debug=True enables auto-reloading and provides detailed error pages
    app.run(debug=True)