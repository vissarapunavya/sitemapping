const ResultsDisplay = ({ results }) => {
  if (!results) {
    return (
      <div className="results-container">
        <p className="placeholder">Select a model, upload an image, and see results here</p>
      </div>
    );
  }

  const avgConfidence = results.predictions.length > 0
    ? (results.predictions.reduce((sum, p) => sum + p.confidence, 0) / results.predictions.length)
    : 0;

  const getTaskIcon = () => {
    return results.task_type === 'detection' ? '🎯' : '✂️';
  };

  return (
    <div className="results-container">
      <h2>📊 Results - {results.model_type === 'soil' ? '🌱 Soil' : '🥕 Vegetable'}</h2>
      
      <div className="task-badge">
        {getTaskIcon()} {results.task_type.toUpperCase()}
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{results.total_detections}</div>
          <div className="metric-label">Total {results.task_type === 'detection' ? 'Detections' : 'Segments'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{Object.keys(results.class_counts).length}</div>
          <div className="metric-label">Unique Classes</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{(avgConfidence * 100).toFixed(1)}%</div>
          <div className="metric-label">Avg Confidence</div>
        </div>
      </div>

      {results.annotated_image && (
        <div className="annotated-image-container">
          <h3>{getTaskIcon()} Annotated Results</h3>
          <img
            src={`data:image/png;base64,${results.annotated_image}`}
            alt="Detection Results"
            className="annotated-image"
          />
        </div>
      )}

      <div className="class-distribution">
        <h3>📈 Class Distribution</h3>
        {Object.entries(results.class_counts).map(([className, count]) => (
          <div key={className} className="class-item">
            <span className="class-name">{className}</span>
            <span className="class-count">{count} {results.task_type === 'detection' ? 'detection(s)' : 'segment(s)'}</span>
          </div>
        ))}
      </div>

      <details className="predictions-details">
        <summary>📋 Detailed Predictions</summary>
        {results.predictions.map((pred, idx) => (
          <div key={idx} className="prediction-item">
            <strong>{results.task_type === 'detection' ? 'Detection' : 'Segment'} {idx + 1}:</strong>
            <p>Class: {pred.class}</p>
            <p>Confidence: {(pred.confidence * 100).toFixed(1)}%</p>
            <p>Type: {pred.type}</p>
            {pred.bbox && (
              <p>BBox: ({pred.bbox.x1.toFixed(0)}, {pred.bbox.y1.toFixed(0)}) to ({pred.bbox.x2.toFixed(0)}, {pred.bbox.y2.toFixed(0)})</p>
            )}
            {pred.mask_available && <p>✅ Mask Available</p>}
          </div>
        ))}
      </details>
    </div>
  );
};

export default ResultsDisplay;
