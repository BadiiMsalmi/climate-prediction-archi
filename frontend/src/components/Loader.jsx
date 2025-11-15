// src/components/Loader.jsx
export default function Loader({ text = "Loading..." }) {
  return (
    <div className="loader">
      <div className="spinner"></div>
      <p>{text}</p>
      <style jsx>{`
        .loader {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          margin: 2rem 0;
        }
        .spinner {
          border: 4px solid rgba(0,0,0,0.1);
          border-left-color: #4f46e5;
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 1s linear infinite;
          margin-bottom: 0.5rem;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
