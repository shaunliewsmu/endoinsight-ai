import ReactPlayer from 'react-player';

function AnalysisResultCard({ result }) {
  console.log('AnalysisResultCard received result:', result);

  if (!result) {
    console.log('Result is null or undefined, returning null');
    return null;
  }

  // Convert relative URLs to absolute URLs
  const getFullUrl = (path) => {
    const fullUrl = `http://localhost:8000${path}`;
    console.log('Converting path to full URL:', { path, fullUrl });
    return fullUrl;
  };

  // Debug video URL
  console.log('Video URL:', {
    original: result.video_url,
    converted: result.video_url ? getFullUrl(result.video_url) : 'No video URL'
  });

  // Debug LLM Analysis
  console.log('LLM Analysis data:', result.llm_analysis);

  // Debug Qualifying Frames
  console.log('Qualifying Frames data:', result.qualifying_frames);

  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body">
        <h2 className="card-title text-2xl mb-6">Analysis Results</h2>
        
        {/* Video Section */}
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4">Analyzed Video</h3>
          {result.video_url ? (
            <div className="relative w-full" style={{ paddingTop: '56.25%' }}>
              <ReactPlayer
                url={getFullUrl(result.video_url)}
                controls
                width="100%"
                height="100%"
                style={{ position: 'absolute', top: 0, left: 0 }}
                config={{
                  file: {
                    attributes: {
                      crossOrigin: "anonymous",
                    },
                    forceVideo: true,
                    forceDASH: false,
                    forceHLS: false,
                  },
                }}
                onError={(e) => {
                  console.error('ReactPlayer error:', e);
                  console.log('Video URL:', getFullUrl(result.video_url));
                }}
                onReady={() => console.log('ReactPlayer ready')}
                onBuffer={() => console.log('Video buffering')}
                onBufferEnd={() => console.log('Video buffering ended')}
                fallback={<div className="text-error">Video playback failed. Please try again.</div>}
              />
            </div>
          ) : (
            <p className="text-error">Video URL not found in response</p>
          )}
        </div>

        {/* LLM Analysis Section */}
        {result.llm_analysis ? (
          <div className="space-y-6 mb-8">
            {/* Glottis Assessment */}
            <div className="card bg-base-100 border">
              <div className="card-body">
                <h3 className="card-title">Glottis Assessment</h3>
                {result.llm_analysis.glottis_assessment ? (
                  <div className="space-y-3">
                    <p><strong>Overall Condition:</strong> {result.llm_analysis.glottis_assessment.overall_condition}</p>
                    <p><strong>Symmetry:</strong> {result.llm_analysis.glottis_assessment.symmetry}</p>
                    <p><strong>Tissue State:</strong> {result.llm_analysis.glottis_assessment.tissue_state}</p>
                    <div>
                      <strong>Abnormalities:</strong>
                      {Array.isArray(result.llm_analysis.glottis_assessment.abnormalities) ? (
                        <ul className="list-disc pl-6 mt-2">
                          {result.llm_analysis.glottis_assessment.abnormalities.map((item, idx) => {
                            console.log('Rendering abnormality:', item);
                            return <li key={idx}>{item}</li>;
                          })}
                        </ul>
                      ) : (
                        <p className="text-error">No abnormalities data found</p>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="text-error">Glottis assessment data not found</p>
                )}
              </div>
            </div>

            {/* Suggested Actions */}
            <div className="card bg-base-100 border">
              <div className="card-body">
                <h3 className="card-title">Suggested Actions</h3>
                {Array.isArray(result.llm_analysis.suggested_actions) ? (
                  <ul className="list-disc pl-6">
                    {result.llm_analysis.suggested_actions.map((action, idx) => {
                      console.log('Rendering action:', action);
                      return <li key={idx} className="mb-2">{action}</li>;
                    })}
                  </ul>
                ) : (
                  <p className="text-error">No suggested actions found</p>
                )}
              </div>
            </div>

            {/* AI Performance */}
            <div className="card bg-base-100 border">
              <div className="card-body">
                <h3 className="card-title">AI Performance Metrics</h3>
                {result.llm_analysis.ai_performance ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(result.llm_analysis.ai_performance).map(([key, value]) => {
                      console.log('Rendering AI metric:', { key, value });
                      const formattedValue = typeof value === 'number' && key !== 'total_frames_analyzed' 
                        ? `${(value * 100).toFixed(1)}%` 
                        : value;
                      return (
                        <div key={key}>
                          <strong>{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}:</strong>
                          {' '}{formattedValue}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-error">AI performance data not found</p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-error">LLM analysis data not found</p>
        )}

        {/* Frames Section */}
        {result.qualifying_frames ? (
          <div>
            <h3 className="text-xl font-semibold mb-4">Analyzed Frames</h3>
            {Array.isArray(result.qualifying_frames.frames) ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {result.qualifying_frames.frames.map((frame, idx) => {
                  console.log('Rendering frame:', { index: idx, frame });
                  return (
                    <div key={idx} className="card bg-base-100 border">
                      <figure className="px-4 pt-4">
                        <img 
                          src={getFullUrl(frame.url)} 
                          alt={`Frame ${idx + 1}`}
                          className="rounded-xl w-full h-48 object-cover"
                          onError={(e) => {
                            console.error('Image load error:', e);
                            e.target.src = 'fallback-image-url.jpg'; // Optional: provide fallback image
                          }}
                          onLoad={() => console.log('Frame image loaded:', idx)}
                        />
                      </figure>
                      <div className="card-body p-4">
                        <h4 className="card-title text-sm">Frame {idx + 1}</h4>
                        <div className="text-sm space-y-1">
                          <p><strong>YOLO Confidence:</strong> {(frame.yolo_detection.confidence * 100).toFixed(1)}%</p>
                          <p><strong>ResNet Prediction:</strong> {frame.resnet_classification.predicted_label}</p>
                          <p><strong>ResNet Confidence:</strong> {(frame.resnet_classification.confidence * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-error">No frames data found</p>
            )}
          </div>
        ) : (
          <p className="text-error">Qualifying frames data not found</p>
        )}
      </div>
    </div>
  );
}

export default AnalysisResultCard;