const API_BASE = import.meta.env.VITE_AUTH_BASE || 'http://localhost:4000';
const ML_BASE = import.meta.env.VITE_ML_BASE || 'http://127.0.0.1:8081';

export async function register(name, email, password) {
  const res = await fetch(`${API_BASE}/api/auth/register`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password })
  });
  if (!res.ok) throw new Error('Registration failed');
  return res.json();
}

export async function login(email, password) {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error('Login failed');
  return res.json();
}

export async function verify(token) {
  const res = await fetch(`${API_BASE}/api/auth/verify`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!res.ok) return { ok: false };
  return res.json();
}

export async function predict(token, file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${ML_BASE}/predict`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form
  });
  if (!res.ok) throw new Error('Prediction failed');
  return res.json();
}
