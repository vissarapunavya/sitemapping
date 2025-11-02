import { useState } from 'react';
import { predictImage } from '../services/api';

const ImageUpload = ({ selectedModel, onResultsReceived }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    if (!selectedModel) {
      setError('Please select a model first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const results = await predictImage(selectedFile, selectedModel);
      onResultsReceived(results);
    } catch (err) {
      setError('Failed to analyze image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getButtonText = () => {
    if (loading) return '🔄 Analyzing...';
    if (selectedModel === 'soil') return '🌱 Detect Soil';
    if (selectedModel === 'vegetable') return '🥕 Segment Vegetables';
    return '🔍 Analyze Image';
  };

  return (
    <div className="upload-container">
      <h2>📤 Upload Image</h2>
      
      <div className="file-input-wrapper">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          id="file-upload"
        />
        <label htmlFor="file-upload" className="file-label">
          Choose Image
        </label>
      </div>

      {preview && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}

      <button
        onClick={handleUpload}
        disabled={!selectedFile || !selectedModel || loading}
        className="upload-button"
      >
        {getButtonText()}
      </button>

      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default ImageUpload;
