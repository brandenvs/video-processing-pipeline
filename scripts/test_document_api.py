import requests
import os
import sys

# API endpoint
url = "http://127.0.0.1:8000/documents/process"

# Get the PDF path from command line or use a default
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    # Default test PDF - replace with an actual PDF on your system
    pdf_path = r"C:\\Users\\liano\\OneDrive\\Documents\\REST-API\\adp-video-pipeline\\input\patrol_report.pdf"

# Check if the file exists
if not os.path.exists(pdf_path):
    print(f"Error: File {pdf_path} not found")
    exit(1)

print(f"Testing with PDF: {pdf_path}")

# Prepare the files and data for the request
files = {
    'file': (os.path.basename(pdf_path), open(pdf_path, 'rb'), 'application/pdf')
}

# Make the request
try:
    # Add a parameter to process immediately instead of in background
    response = requests.post(url, files=files, data={'process_immediately': 'true'})
    
    # Print the response
    print(f"Status code: {response.status_code}")
    print("Response:")
    print(response.json())
    
except Exception as e:
    print(f"Error: {e}")
finally:
    # Close the file
    files['file'][1].close()

# Test listing documents
try:
    list_url = "http://127.0.0.1:8000/documents/list"
    list_response = requests.get(list_url)
    print("\nDocument List:")
    print(list_response.json())
except Exception as e:
    print(f"Error listing documents: {e}")