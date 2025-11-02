import { useState, useEffect } from 'react';
import { getAvailableModels } from '../services/api';

const ModelSelector = ({ selectedModel, onModelChange }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      const availableModels = await getAvailableModels();
      setModels(availableModels);
      setLoading(false);
      
      // Set default model if not selected
      if (!selectedModel && availableModels.length > 0) {
        onModelChange(availableModels[0].id);
      }
    };
    fetchModels();
  }, []);

  return (
    <div className="model-selector">
      <h3>🤖 Select Model</h3>
      {loading ? (
        <p>Loading models...</p>
      ) : (
        <div className="model-options">
          {models.map((model) => (
            <div
              key={model.id}
              className={`model-card ${selectedModel === model.id ? 'selected' : ''} ${!model.available ? 'disabled' : ''}`}
              onClick={() => model.available && onModelChange(model.id)}
            >
              <h4>{model.name}</h4>
              <p className="model-type">{model.type}</p>
              <p className="model-classes">
                {model.classes.length} classes
              </p>
              {!model.available && <span className="unavailable">Unavailable</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
