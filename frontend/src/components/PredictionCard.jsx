// src/components/PredictionCard.jsx
import { useEffect, useState } from "react";
import { getNextPrediction } from "../api/predictions";
import Loader from "./Loader";

export default function PredictionCard() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPrediction() {
      try {
        const data = await getNextPrediction();
        setPrediction(data.predicted_temperature_next_hour);
      } catch (err) {
        console.error("Error fetching prediction:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchPrediction();
  }, []);

  return (
    <div className="card">
      <h2>Next Hour Prediction</h2>
      {loading ? <Loader /> : prediction !== null ? <p className="temp">{prediction}°C</p> : <p>No data</p>}

      <style jsx>{`
        .card {
          padding: 1rem;
          margin-bottom: 1.5rem;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: #f9f9f9;
          text-align: center;
        }
        .temp {
          font-size: 2rem;
          color: #4f46e5;
          margin-top: 0.5rem;
        }
      `}</style>
    </div>
  );
}
