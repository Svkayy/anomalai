# Anomalai: Advanced Media Analysis Platform 

Anomalai is a powerful Python-based platform for advanced image and video analysis. It leverages state-of-the-art machine learning models to perform detailed media segmentation, automatic object labeling, and content safety classification. 

## Features

* **AI-Powered Segmentation**: Utilizes the **Segment Anything Model (SAM2)** to accurately identify and mask distinct objects and regions in images and video frames.
* **Automatic Labeling with CLIP**: Employs **CLIP (Contrastive Language-Image Pre-training)** for zero-shot classification of segmented regions, providing descriptive labels and confidence scores without model training.
* **Comprehensive Video Analysis**: Processes video files to perform frame-by-frame analysis, extending image-based insights to video content.
* **Content Safety Classifier**: Includes a dedicated module to assess media for safety, helping to moderate and flag potentially sensitive content.
* **Retrieval-Augmented Generation (RAG)**: Integrates a RAG system to enable intelligent querying and interaction with the analysis results.
* **Scalable Backend**: Uses **Supabase** for database management, handling file storage and metadata with ease.
* **Optimized Performance**: Engineered with parallel processing to handle demanding analysis tasks efficiently.

***

## Getting Started

Follow these instructions to set up and run the Anomalai platform on your local machine.

### Prerequisites

* [Conda/Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) installed on your system.
* Access to a [Supabase](https://supabase.com/) project.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/svkayy-anomalai.git](https://github.com/your-username/svkayy-anomalai.git)
cd svkayy-anomalai
```

### 2. Set Up the Environment

Create and activate the Conda environment using the provided `environment.yml` file. This will install all the necessary dependencies.

```bash
conda env create -f environment.yml
conda activate myvenv
```

Alternatively, for a lightweight setup, you can use `pip`:

```bash
pip install -r requirements.txt
```

### 3. Configure the Database

1.  Navigate to your Supabase project's SQL Editor.
2.  Open the `setup_database.sql` file from this repository.
3.  Copy its contents and run it in the Supabase SQL Editor to create the required tables and storage buckets.

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your Supabase credentials. The `supabase_database.py` and other modules will use these variables to connect to your backend.

```env
SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL"
SUPABASE_KEY="YOUR_SUPABASE_SERVICE_ROLE_KEY"
```

***

## Usage

The platform's core logic is initiated via `app.py`, which orchestrates the analysis pipeline. To begin processing, run the script from your terminal:

```bash
python app.py
```

The Flask app will then run at http://127.0.0.1:5000/

The script will interact with the other modules to perform tasks such as video processing, safety classification, and database updates. For specific API interactions, such as image segmentation, refer to the detailed guides below.

***

