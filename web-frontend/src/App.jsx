import React, { useState, useEffect } from 'react'
import { login, register, verify, predict } from './api'

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '')
  const [mode, setMode] = useState('login')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')

  useEffect(() => {
    if (!token) return
    verify(token).then(v => {
      if (!v.ok) {
        localStorage.removeItem('token')
        setToken('')
      }
    }).catch(() => {
      localStorage.removeItem('token'); setToken('')
    })
  }, [])

  async function onLogin(e) {
    e.preventDefault()
    try {
      const data = await login(email, password)
      localStorage.setItem('token', data.token)
      setToken(data.token)
    } catch (e) {
      alert('Login failed')
    }
  }

  async function onRegister(e) {
    e.preventDefault()
    try {
      const data = await register(name, email, password)
      localStorage.setItem('token', data.token)
      setToken(data.token)
    } catch (e) {
      alert('Registration failed')
    }
  }

  async function onPredict(e) {
    e.preventDefault()
    if (!file) return
    try {
      const res = await predict(token, file)
      setResult(res.hgl)
      setStatus(res.status)
    } catch (e) {
      alert('Prediction failed')
    }
  }

  if (!token) {
    return (
      <div style={{ maxWidth: 420, margin: '40px auto', fontFamily: 'system-ui' }}>
        <h2>Anemia App – {mode === 'login' ? 'Login' : 'Register'}</h2>
        <form onSubmit={mode === 'login' ? onLogin : onRegister}>
          {mode === 'register' && (
            <div>
              <label>Name</label><br />
              <input value={name} onChange={e => setName(e.target.value)} required style={{ width: '100%' }} />
            </div>
          )}
          <div style={{ marginTop: 12 }}>
            <label>Email</label><br />
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required style={{ width: '100%' }} />
          </div>
          <div style={{ marginTop: 12 }}>
            <label>Password</label><br />
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required style={{ width: '100%' }} />
          </div>
          <button style={{ marginTop: 16 }} type="submit">{mode === 'login' ? 'Login' : 'Create account'}</button>
        </form>
        <p style={{ marginTop: 12 }}>
          {mode === 'login' ? (
            <>Don’t have an account? <button onClick={() => setMode('register')}>Register</button></>
          ) : (
            <>Already have an account? <button onClick={() => setMode('login')}>Login</button></>
          )}
        </p>
      </div>
    )
  }

  return (
    <div style={{ maxWidth: 640, margin: '40px auto', fontFamily: 'system-ui' }}>
      <h2>Anemia App – Dashboard</h2>
      <button onClick={() => { localStorage.removeItem('token'); setToken('') }}>Logout</button>
      <form onSubmit={onPredict} style={{ marginTop: 20 }}>
        <input type="file" accept="image/*" onChange={e => setFile(e.target.files?.[0] || null)} />
        <button type="submit" disabled={!file} style={{ marginLeft: 12 }}>Predict</button>
      </form>
      {result && (
        <div style={{ marginTop: 20 }}>
          <div>Hemoglobin: {result}</div>
          <div>Status: {status}</div>
        </div>
      )}
    </div>
  )
}
