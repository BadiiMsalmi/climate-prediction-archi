// src/pages/Home.jsx
import { useEffect, useState } from "react";
import PredictionCard from "../components/PredictionCard";
import HistoryChart from "../components/HistoryChart";
import Loader from "../components/Loader";

export default function Home() {
  const [loading, setLoading] = useState(true);

  // optional: delay to show loader until both predictions and history are ready
  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="container">
      <h1>Temperature Prediction Dashboard</h1>
      {loading ? (
        <Loader />
      ) : (
        <>
          <PredictionCard />
          <HistoryChart />
        </>
      )}

      <style jsx>{`
        .container {
          max-width: 800px;
          margin: 2rem auto;
          font-family: sans-serif;
        }
        h1 {
          text-align: center;
          margin-bottom: 2rem;
          color: #4f46e5;
        }
      `}</style>
    </div>
  );
}
