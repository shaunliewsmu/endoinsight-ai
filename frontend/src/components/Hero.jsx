import AnimatedDrawing from "./AnimatedDrawing";
import TypingText from "./TypingText";
import curveline from "../assets/curveline.svg";
const Hero = () => {
    return (
        <div className="hero bg-base-100 rounded-box mb-8 "
        style={{
    backgroundImage: `url(${curveline})`,
  }}>
        <div className="hero-content text-center">
            
          <div className="max-w-4xl">
            <AnimatedDrawing word = "AI H&N Cancer Screening" class = "p-4"/>
            <TypingText text="AI-Powered Laryngeal Cancer Detection and Triage Analysis System" speed={80} fontSize="text-lg" color="text-blue-800" fontStyle="italic" />
            <p className="text-lg mb-4 italic font-bold">
              Revolutionizing Head and Neck Cancer Screening in Low-Resource Settings through Advanced AI Technology
            </p>
            <p className="text-base">
            AI Head and Neck Cancer Screen combines advanced computer vision and machine learning to analyze laryngeal 
            endoscopy videos with expert-level precision. 
            Our system provides rapid screening support and clinical insights for healthcare 
            providers in resource-limited settings, 
            enabling earlier detection of head and neck cancers through automated analysis.
            </p>
          </div>
        </div>
      </div>
    );
  };
  
  export default Hero;