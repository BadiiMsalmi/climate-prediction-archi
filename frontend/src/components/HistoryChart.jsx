// src/components/HistoryChart.jsx
import { useEffect, useRef, useState } from "react";
import { getOldPredictions } from "../api/predictions";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  TimeScale,
  Title,
  Tooltip
} from "chart.js";
import Loader from "./Loader";

// Register all Chart.js components
Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  TimeScale,
  Title,
  Tooltip
);

export default function HistoryChart() {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await getOldPredictions();
        const labels = data.map(item => item.pred_ts);
        const temps = data.map(item => item.predicted_temperature);

        // Destroy previous chart instance if exists
        if (chartRef.current) {
          chartRef.current.destroy();
        }

        const ctx = canvasRef.current.getContext("2d");

        chartRef.current = new Chart(ctx, {
          type: "line",
          data: {
            labels,
            datasets: [
              {
                label: "Predicted Temperature (°C)",
                data: temps,
                borderColor: "#4f46e5",
                backgroundColor: "rgba(79, 70, 229, 0.2)",
                borderWidth: 2,
                tension: 0.3, // smooth curve
                pointRadius: 3
              }
            ]
          },
          options: {
            responsive: true,
            plugins: {
              title: {
                display: true,
                text: "Prediction History"
              },
              tooltip: {
                mode: "index",
                intersect: false
              }
            },
            scales: {
              x: {
                type: "category",
                title: {
                  display: true,
                  text: "Timestamp"
                }
              },
              y: {
                title: {
                  display: true,
                  text: "Temperature (°C)"
                }
              }
            }
          }
        });

        setLoading(false);
      } catch (error) {
        console.error("Failed to load chart data:", error);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="chart-wrapper">
      {loading ? (
        <Loader text="Loading prediction history..." />
      ) : (
        <canvas ref={canvasRef}></canvas>
      )}
      <style jsx>{`
        .chart-wrapper {
          margin-top: 2rem;
          padding: 1rem;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: #f9f9f9;
        }
      `}</style>
    </div>
  );
}
