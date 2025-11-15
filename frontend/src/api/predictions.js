const API_URL = "http://localhost:8000";

export async function getNextPrediction() {
  const res = await fetch(`${API_URL}/predict`);
  return res.json();
}

export async function getOldPredictions() {
  const res = await fetch(`${API_URL}/predictions`);
  return res.json();
}
