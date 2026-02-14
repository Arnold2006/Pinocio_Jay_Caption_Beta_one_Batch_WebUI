JoyCaption - Advanced Image Captioning Tool
Show Image
This is a simple Gradio GUI for the JoyCaption model with enhanced usability and stability.
Installation
Prerequisites

Python 3.8+
CUDA-capable GPU (recommended)
At least 24GB VRAM recommended

Setup

Clone this repository:

bash   git clone https://github.com/yourusername/joy-caption.git
   cd joy-caption/gradio-app

Set up a Python virtual environment (optional but recommended):

bash   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:

bash   pip install -r requirements.txt

Run the application:

bash   python app.py

Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860)

Optional: Install Liger Kernel
For faster inference, you can install the Liger Kernel: https://github.com/linkedin/Liger-Kernel
Usage
Single Image Captioning

Upload an image using the left-hand panel
Select a caption type and desired caption length
(Optional) Open "Extra Options" and select any additional parameters
(Optional) Adjust generation settings like temperature and top-p values
Click "Caption" to generate
The generated caption will appear in the output box and can be copied or edited

Batch Processing

Switch to the "Batch Processing" tab
Upload multiple images (PNG/JPEG/WEBP)
Specify an output folder path where caption .txt files will be saved (the folder must already exist)
Set the DataLoader Workers (CPU processes) and Batch Size based on your system capabilities
Click "Start Batch Process"
Caption files will be saved as .txt files in your specified output folder (one .txt file per image with matching filenames)

Caption Types
ModeDescriptionDescriptiveFormal, detailed prose descriptionDescriptive (Casual)Similar to Descriptive but with a friendlier, conversational toneStraightforwardObjective, no fluff, and more succinct than DescriptiveStable Diffusion PromptReverse-engineers a prompt for Stable Diffusion modelsMidJourneySimilar to the above but tuned for MidJourney's prompt styleDanbooru tag listComma-separated tags following Danbooru conventionse621 tag listAlphabetical, namespaced tags in e621 styleRule34 tag listRule34 style alphabetical tag dumpBooru-like tag listLooser tag list for general labelingArt CriticArt-historical commentary on composition, symbolism, style, etc.Product ListingMarketing copy as if selling the depicted objectSocial Media PostCatchy caption for social media platforms

Note on Booru modes: They're optimized for anime-style/illustration imagery; accuracy may decrease with real-world photographs or abstract artwork.

Features

BFloat16 precision for optimal quality and performance
Batch processing with customizable output locations
Flexible caption styles from formal descriptions to social media posts
Adjustable generation parameters (temperature, top-p, max tokens)
Clean, accessible UI with improved visibility and layout
Progress tracking for batch operations
