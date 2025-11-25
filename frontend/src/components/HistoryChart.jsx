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
  const [data, setData] = useState(null);

  // Step 1 — fetch the data ONLY
  useEffect(() => {
    getOldPredictions()
      .then(res => {
        setData(res);   // this triggers the DOM to re-render with the canvas
      })
      .catch(err => console.error("Failed to load chart data:", err));
  }, []);

  // Step 2 — chart is created ONLY AFTER canvas is rendered AND data exists
  useEffect(() => {
    if (!data || !canvasRef.current) return; // WAIT until canvas exists

    const labels = data.map(item => item.pred_ts);
    const temps = data.map(item => item.predicted_temperature);

    if (chartRef.current) chartRef.current.destroy();

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
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: "Prediction History"
          }
        },
        scales: {
          x: {
            type: "category"
          }
        }
      }
    });

  }, [data]); // <-- runs ONLY after data is fetched AND canvas is ready

  return (
    <div className="chart-wrapper">
      {!data ? <Loader text="Loading prediction history..." /> : <canvas ref={canvasRef} />}
    </div>
  );
}
