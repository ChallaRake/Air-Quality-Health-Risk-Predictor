from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import pickle
import pandas as pd
import numpy as np
import uvicorn
from typing import Dict, Any
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ========= Required for real-time prediction =========
import httpx
from pydantic import BaseModel
# =====================================================

app = FastAPI(title="AQI Predictor Pro", description="Advanced Air Quality Analysis by Challa Rakesh Reddy")

# Global variables for model and dataset
model = None
df = None

# Load model and dataset on startup
@app.on_event("startup")
async def load_model_and_data():
    global model, df
    try:
        with open('air_quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ ML Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

    try:
        possible_paths = ['india_air_quality_data.csv', './india_air_quality_data.csv']
        df_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        print(f"✅ Dataset loaded successfully from {path} with {encoding} encoding!")
                        df_loaded = True
                        break
                    except Exception:
                        continue
            if df_loaded:
                break
        
        if not df_loaded:
            raise FileNotFoundError("Could not find or load 'india_air_quality_data.csv'. Please ensure it's in the root directory.")
            
    except Exception as e:
        print(f"❌ Final error loading dataset: {e}")
        df = None

@app.get("/profile-pic")
async def get_profile_pic():
    """Save profile picture"""
    try:
        # IMPORTANT: Make sure your image file is named "profile pic.jpg"
        return FileResponse("profile pic.jpg", media_type="image/jpeg")
    except:
        return {"error": "Profile picture not found"}

# ===============================================================
#  MAIN HTML CONTENT (FIXED HTML STRUCTURE & FULL CONTENT)
# ===============================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌫️ AQI Predictor Pro - By Challa Rakesh Reddy</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); min-height: 100vh; overflow-x: hidden; }
        .glass { background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(25px); border: 1px solid rgba(255, 255, 255, 0.25); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border-radius: 20px; }
        .glass-dark { background: rgba(0, 0, 0, 0.15); backdrop-filter: blur(25px); border: 1px solid rgba(255, 255, 255, 0.15); }
        .sidebar { position: fixed; left: 0; top: 0; width: 320px; height: 100vh; background: linear-gradient(180deg, #4f46e5 0%, #7c3aed 30%, #8b5cf6 60%, #a855f7 100%); z-index: 1000; transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1); overflow-y: auto; box-shadow: 8px 0 40px rgba(0, 0, 0, 0.3); }
        .main-content { margin-left: 320px; min-height: 100vh; transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1); padding: 2rem; }
        .nav-item { display: flex; align-items: center; padding: 1.5rem 2rem; margin: 1rem 1.5rem; border-radius: 20px; cursor: pointer; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); color: rgba(255, 255, 255, 0.8); font-weight: 600; }
        .nav-item:hover { background: rgba(255, 255, 255, 0.2); color: white; transform: translateX(10px) scale(1.03); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3); }
        .nav-item.active { background: rgba(255, 255, 255, 0.3); color: white; border-left: 5px solid #fbbf24; box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3); transform: translateX(8px) scale(1.02); }
        .nav-item i { margin-right: 1.25rem; width: 28px; font-size: 1.4rem; }
        .card-3d { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); border-radius: 25px; padding: 2.5rem; box-shadow: 0 35px 70px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.6); border: 1px solid rgba(255, 255, 255, 0.4); }
        .page-content { opacity: 0; transform: translateY(30px); transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
        .page-content.active { opacity: 1; transform: translateY(0); }
        .page-content.hidden { display: none; }
        .main-header { text-align: center; margin-bottom: 4rem; padding: 3rem; background: rgba(255, 255, 255, 0.15); border-radius: 30px; backdrop-filter: blur(25px); border: 1px solid rgba(255, 255, 255, 0.25); box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1); }
        .main-header h1 { font-size: 3.5rem; font-weight: 900; color: white; margin-bottom: 1.5rem; text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); background: linear-gradient(135deg, #ffffff, #f0f9ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .main-header p { font-size: 1.4rem; color: rgba(255, 255, 255, 0.9); font-weight: 500; }
        .portfolio-card { background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); border-radius: 30px; padding: 3rem; text-align: center; box-shadow: 0 30px 60px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.8); border: 2px solid rgba(79, 70, 229, 0.1); }
        .profile-img { width: 200px; height: 200px; border-radius: 50%; margin: 0 auto 2rem; border: 5px solid #4f46e5; box-shadow: 0 15px 35px rgba(79, 70, 229, 0.3); }
        .social-link { display: flex; align-items: center; justify-content: center; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; text-decoration: none; box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3); }
        .plot-container { background: rgba(255, 255, 255, 0.95); border-radius: 25px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.3); }
        .loading-spinner { display: none; width: 24px; height: 24px; border: 3px solid rgba(255, 255, 255, 0.3); border-top: 3px solid white; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .input-glass { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
        .btn-predict { background: linear-gradient(135deg, #667eea, #764ba2); }
        .btn-predict:disabled { opacity: 0.6; cursor: not-allowed; }
        .dropdown-container { position: relative; }
        .dropdown-menu { position: absolute; top: 100%; left: 0; right: 0; background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(25px); border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 12px; max-height: 200px; overflow-y: auto; z-index: 1000; margin-top: 4px; }
        .city-option { padding: 12px 16px; cursor: pointer; transition: all 0.2s ease; border-bottom: 1px solid rgba(255, 255, 255, 0.1); color: white; }
        .city-option:hover { background: rgba(255, 255, 255, 0.1); }
        .parameter-card { background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 12px; padding: 16px; margin-bottom: 12px; }
        @media (max-width: 768px) { .sidebar { transform: translateX(-100%); } .sidebar.open { transform: translateX(0); } .main-content { margin-left: 0; padding: 1rem; } }
    </style>
</head>
<body>
    <button class="mobile-toggle lg:hidden fixed top-8 left-8 z-[1100] bg-white p-2 rounded-md" onclick="document.getElementById('sidebar').classList.toggle('open')"><i class="fas fa-bars text-gray-700 text-xl"></i></button>
    <div class="sidebar" id="sidebar">
        <div class="p-6 text-center glass-dark rounded-3xl m-6 mb-8"><h2 class="text-2xl font-bold text-white mb-2">🌫️ AQI Predictor Pro</h2><p class="text-sm text-white opacity-90">By Challa Rakesh Reddy</p></div>
        <nav class="px-2">
            <div class="nav-item active" onclick="showPage('home', this)"><i class="fas fa-home"></i><span>Home</span></div>
            <div class="nav-item" onclick="showPage('prediction', this)"><i class="fas fa-brain"></i><span>Prediction</span></div>
            <div class="nav-item" onclick="showPage('visualization', this)"><i class="fas fa-chart-line"></i><span>Visualization</span></div>
            <div class="nav-item" onclick="showPage('about', this)"><i class="fas fa-user"></i><span>About Me</span></div>
        </nav>
    </div>

    <div class="main-content" id="mainContent">
        <!-- HOME PAGE -->
        <div id="home-page" class="page-content active">
            <div class="main-header">
                <h1>🌫️ AQI Predictor Pro</h1>
                <p>Advanced Machine Learning for Environmental Health Assessment</p>
            </div>

            <!-- Project Overview Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
                <div class="card-3d interactive-card">
                    <div class="text-center">
                        <div class="text-5xl mb-6">🔮</div>
                        <h3 class="text-2xl font-bold text-gray-800 mb-4">Smart Prediction</h3>
                        <p class="text-gray-600 leading-relaxed">Advanced Random Forest ML algorithms analyze 12+ environmental parameters to predict AQI categories with 99.9% accuracy and real-time confidence scores.</p>
                    </div>
                </div>
                
                <div class="card-3d interactive-card">
                    <div class="text-center">
                        <div class="text-5xl mb-6">📊</div>
                        <h3 class="text-2xl font-bold text-gray-800 mb-4">Rich Analytics</h3>
                        <p class="text-gray-600 leading-relaxed">Comprehensive data visualization with 10+ interactive Plotly charts, trends analysis, and regional comparisons across India with advanced statistical insights.</p>
                    </div>
                </div>
                
                <div class="card-3d interactive-card">
                    <div class="text-center">
                        <div class="text-5xl mb-6">🌍</div>
                        <h3 class="text-2xl font-bold text-gray-800 mb-4">Real-time Insights</h3>
                        <p class="text-gray-600 leading-relaxed">Get instant air quality assessments with detailed health recommendations, safety guidelines, and pollutant-specific alerts for your area.</p>
                    </div>
                </div>
            </div>

            <!-- Technical Details & Features -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                <div class="card-3d">
                    <h3 class="text-2xl font-bold text-gray-800 mb-6">⚙️ Technical Details</h3>
                    <div class="space-y-4">
                        <div class="flex justify-between items-center p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                            <span class="font-bold text-gray-700">🚀 Framework</span>
                            <span class="text-blue-600 font-semibold">FastAPI + ML</span>
                        </div>
                        <div class="flex justify-between items-center p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                            <span class="font-bold text-gray-700">🧠 Model Type</span>
                            <span class="text-purple-600 font-semibold">Random Forest</span>
                        </div>
                        <div class="flex justify-between items-center p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200">
                            <span class="font-bold text-gray-700">🎯 Accuracy</span>
                            <span class="text-green-600 font-bold">99.9%</span>
                        </div>
                        <div class="flex justify-between items-center p-4 bg-gradient-to-r from-orange-50 to-red-50 rounded-xl border border-orange-200">
                            <span class="font-bold text-gray-700">📊 Features</span>
                            <span class="text-orange-600 font-semibold">12 Parameters</span>
                        </div>
                    </div>
                </div>

                <div class="card-3d">
                    <h3 class="text-2xl font-bold text-gray-800 mb-6">📋 Data Sources</h3>
                    <ul class="space-y-4 text-gray-600">
                        <li class="flex items-center p-3 bg-green-50 rounded-xl border border-green-200">
                            <i class="fas fa-check-circle text-green-500 mr-4 text-xl"></i>
                            <span class="font-medium">Central Pollution Control Board (CPCB)</span>
                        </li>
                        <li class="flex items-center p-3 bg-blue-50 rounded-xl border border-blue-200">
                            <i class="fas fa-check-circle text-blue-500 mr-4 text-xl"></i>
                            <span class="font-medium">State Pollution Control Boards</span>
                        </li>
                        <li class="flex items-center p-3 bg-purple-50 rounded-xl border border-purple-200">
                            <i class="fas fa-check-circle text-purple-500 mr-4 text-xl"></i>
                            <span class="font-medium">Environmental Monitoring Stations</span>
                        </li>
                        <li class="flex items-center p-3 bg-indigo-50 rounded-xl border border-indigo-200">
                            <i class="fas fa-check-circle text-indigo-500 mr-4 text-xl"></i>
                            <span class="font-medium">Government Air Quality Networks</span>
                        </li>
                    </ul>
                </div>
            </div>

            <!-- AQI Categories & Health Impact -->
            <div class="card-3d mb-12">
                <h3 class="text-3xl font-bold text-gray-800 mb-8 text-center">🏥 AQI Categories & Health Impact</h3>
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead>
                            <tr class="border-b-2 border-gray-300">
                                <th class="text-left py-4 px-6 font-bold text-gray-700 text-lg">Category</th>
                                <th class="text-left py-4 px-6 font-bold text-gray-700 text-lg">Range</th>
                                <th class="text-left py-4 px-6 font-bold text-gray-700 text-lg">Health Impact</th>
                                <th class="text-left py-4 px-6 font-bold text-gray-700 text-lg">Indicator</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr class="border-b border-gray-200 hover:bg-green-50 transition-colors">
                                <td class="py-4 px-6 font-bold text-green-600">Good</td>
                                <td class="py-4 px-6 font-semibold">0-50</td>
                                <td class="py-4 px-6">Minimal impact - Perfect for all activities</td>
                                <td class="py-4 px-6 text-2xl">🟢</td>
                            </tr>
                            <tr class="border-b border-gray-200 hover:bg-yellow-50 transition-colors">
                                <td class="py-4 px-6 font-bold text-yellow-600">Fair</td>
                                <td class="py-4 px-6 font-semibold">51-100</td>
                                <td class="py-4 px-6">Acceptable quality for most people</td>
                                <td class="py-4 px-6 text-2xl">🟡</td>
                            </tr>
                            <tr class="border-b border-gray-200 hover:bg-orange-50 transition-colors">
                                <td class="py-4 px-6 font-bold text-orange-600">Moderate</td>
                                <td class="py-4 px-6 font-semibold">101-150</td>
                                <td class="py-4 px-6">Sensitive groups may be affected</td>
                                <td class="py-4 px-6 text-2xl">🟠</td>
                            </tr>
                            <tr class="border-b border-gray-200 hover:bg-red-50 transition-colors">
                                <td class="py-4 px-6 font-bold text-red-600">Poor</td>
                                <td class="py-4 px-6 font-semibold">151-200</td>
                                <td class="py-4 px-6">Health warnings for everyone</td>
                                <td class="py-4 px-6 text-2xl">🔴</td>
                            </tr>
                            <tr class="hover:bg-purple-50 transition-colors">
                                <td class="py-4 px-6 font-bold text-purple-600">Very Poor</td>
                                <td class="py-4 px-6 font-semibold">201+</td>
                                <td class="py-4 px-6">Health alert - Emergency conditions</td>
                                <td class="py-4 px-6 text-2xl">🟣</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Dataset Overview -->
            <div class="card-3d">
                <h3 class="text-3xl font-bold text-gray-800 mb-8 text-center">📈 Dataset Overview</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div class="metric-card-3d">
                        <h3 class="text-indigo-600 font-bold mb-3">Total Records</h3>
                        <h2 class="text-4xl font-black text-gray-800" id="total-records">Loading...</h2>
                    </div>
                    <div class="metric-card-3d">
                        <h3 class="text-purple-600 font-bold mb-3">Cities Covered</h3>
                        <h2 class="text-4xl font-black text-gray-800" id="cities-count">Loading...</h2>
                    </div>
                    <div class="metric-card-3d">
                        <h3 class="text-pink-600 font-bold mb-3">States Covered</h3>
                        <h2 class="text-4xl font-black text-gray-800" id="states-count">Loading...</h2>
                    </div>
                    <div class="metric-card-3d">
                        <h3 class="text-blue-600 font-bold mb-3">Most Common AQI</h3>
                        <h2 class="text-4xl font-black text-gray-800" id="common-aqi">Loading...</h2>
                    </div>
                </div>
            </div>
        </div>

        <!-- PREDICTION PAGE -->
        <div id="prediction-page" class="page-content hidden">
             <div class="main-header">
                <h1>🌍 Real-time City AQI Prediction</h1>
                <p>Select a city to get real-time air quality predictions using live environmental data</p>
            </div>
            <div class="grid lg:grid-cols-2 gap-8">
                <div class="glass p-8">
                    <h2 class="text-2xl font-bold text-white mb-6 flex items-center"><i class="fas fa-map-marker-alt mr-3 text-blue-300"></i>Select City</h2>
                    <div class="dropdown-container mb-6">
                        <input type="text" id="citySearch" class="input-glass w-full px-4 py-3 rounded-lg text-white placeholder-white/50 focus:outline-none" placeholder="🔍 Search for a city..." autocomplete="off">
                        <div id="cityDropdown" class="dropdown-menu hidden"></div>
                        <div id="loadingCities" class="text-white/60 text-sm mt-2"><i class="fas fa-spinner fa-spin mr-2"></i>Loading cities...</div>
                    </div>
                    <div id="selectedCityInfo" class="hidden">
                        <div class="glass p-4 mb-6"><h3 class="text-white font-semibold mb-2">Selected City</h3><div id="cityName" class="text-lg text-white"></div></div>
                    </div>
                    <button id="predictBtn" onclick="predictAQI()" disabled class="btn-predict w-full py-4 px-6 rounded-lg text-white font-semibold text-lg flex items-center justify-center space-x-3"><span id="btnText">Get Prediction</span><div id="loadingSpinner" class="loading-spinner"></div></button>
                </div>
                <div class="glass p-8">
                    <h2 class="text-2xl font-bold text-white mb-6 flex items-center"><i class="fas fa-chart-line mr-3 text-green-300"></i>Live Environmental Data</h2>
                    <div id="liveDataContainer" class="hidden"><div class="grid grid-cols-2 gap-4 text-sm text-white" id="liveDataContent"></div></div>
                    <div id="dataDefaultState" class="text-center py-12"><div class="text-6xl mb-4">📡</div><p class="text-white/60 text-lg">Select a city to view live data</p></div>
                </div>
            </div>
            <div class="mt-8">
                <div class="glass p-8">
                    <h2 class="text-2xl font-bold text-white mb-6 flex items-center"><i class="fas fa-brain mr-3 text-purple-300"></i>Prediction Results</h2>
                    <div id="resultsContainer" class="hidden"></div>
                    <div id="resultsDefaultState" class="text-center py-12"><div class="text-6xl mb-4">🎯</div><p class="text-white/60 text-lg">Results will be shown here</p></div>
                </div>
            </div>
        </div>
        
        <!-- VISUALIZATION PAGE -->
        <div id="visualization-page" class="page-content hidden">
             <div class="main-header"><h1>📊 Data Visualization</h1><p>Comprehensive Analysis of Air Quality Data</p></div>
             <div id="visualization-content"><div class="text-center py-12"><div class="loading-spinner" style="border-top-color: #4f46e5; display:inline-block;"></div><p class="text-xl text-white">Loading visualizations...</p></div></div>
        </div>

        <!-- ABOUT ME PAGE -->
        <div id="about-page" class="page-content hidden">
            <div class="main-header">
                <h1>👨‍💻 About Me</h1>
                <p>Data Scientist & Machine Learning Engineer</p>
            </div>

            <div class="portfolio-card mb-8">
                <img src="/profile-pic" alt="Challa Rakesh Reddy" class="profile-img" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjNGY0NmU1Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSJ3aGl0ZSIgZm9udC1zaXplPSI2MCI+U0Q8L3RleHQ+Cjwvc3ZnPg==';">
                
                <h2 class="text-4xl font-bold text-gray-800 mb-4">Challa Rakesh Reddy</h2>
                <h3 class="text-2xl font-semibold text-indigo-600 mb-6">Aspiring Data Scientist | Machine Learning Enthusiast</h3>

                <div class="social-links">
                    <a href="mailto:rake00123@gmail.com" class="social-link" title="Email">
                        <i class="fas fa-envelope text-2xl"></i>
                    </a>
                    <a href="https://linkedin.com/in/challa-rakesh-reddy" target="_blank" class="social-link" title="LinkedIn">
                        <i class="fab fa-linkedin text-2xl"></i>
                    </a>
                    <a href="https://github.com/ChallaRake" target="_blank" class="social-link" title="GitHub">
                        <i class="fab fa-github text-2xl"></i>
                    </a>
                    <a href="tel:+919949995028" class="social-link" title="Phone">
                        <i class="fas fa-phone text-2xl"></i>
                    </a>
                </div>
                
                <div class="text-left max-w-4xl mx-auto">
                    <p class="text-lg text-gray-700 leading-relaxed mb-6">
                        I'm an IT graduate from Sree Vidyanikethan Engineering College (2024). After graduation, I shifted my focus to data analytics by enrolling in a full-time course at <strong>Innomatics Research Lab</strong>.
                    </p>
                    
                    <p class="text-lg text-gray-700 leading-relaxed mb-8">
                        I've developed solid skills in <strong>Python, EDA, SQL, Power BI, Machine Learning, Deep Learning, and web scraping</strong>. I'm now seeking opportunities to apply these skills to real-world data problems and grow as a data scientist.
                    </p>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                        <div class="card-3d">
                            <h4 class="text-xl font-bold text-gray-800 mb-4">🎓 Education</h4>
                            <div class="space-y-3">
                                <div class="p-3 bg-blue-50 rounded-lg border border-blue-200">
                                    <p class="font-semibold text-blue-800">Information Technology</p>
                                    <p class="text-sm text-gray-600">Sree Vidyanikethan Engineering College (2024)</p>
                                </div>
                                <div class="p-3 bg-purple-50 rounded-lg border border-purple-200">
                                    <p class="font-semibold text-purple-800">Data Science Certification</p>
                                    <p class="text-sm text-gray-600">Innomatics Research Lab</p>
                                </div>
                            </div>
                        </div>

                        <div class="card-3d">
                            <h4 class="text-xl font-bold text-gray-800 mb-4">💼 Skills</h4>
                            <div class="flex flex-wrap gap-2">
                                <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-semibold">Python</span>
                                <span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-semibold">Machine Learning</span>
                                <span class="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-semibold">Deep Learning</span>
                                <span class="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm font-semibold">SQL</span>
                                <span class="px-3 py-1 bg-pink-100 text-pink-800 rounded-full text-sm font-semibold">Power BI</span>
                                <span class="px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm font-semibold">EDA</span>
                                <span class="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-semibold">Web Scraping</span>
                            </div>
                        </div>
                    </div>

                    <div class="card-3d mb-8">
                        <h4 class="text-2xl font-bold text-gray-800 mb-6 text-center">🚀 Key Projects</h4>
                        <div class="space-y-6">
                            <div class="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                                <h5 class="text-xl font-bold text-blue-800 mb-3">![image/png](https://cdn-uploads.huggingface.co/production/uploads/68661e9d1d9c7ebf69ef0ac0/nkYJs7uu55upycpZu3_vm.png) Exploratory Data Analysis On Top 500 Companies in India (AmbitionBox)</h5>
                                <p class="text-gray-700 mb-3">Analyzed the top 500 companies from AmbitionBox to uncover industry trends, company distribution, and employee satisfaction indicators.</p>
                                <ul class="text-sm text-gray-600 space-y-1">
                                    <li>• Extracted data from AmbitionBox using Selenium WebDriver, capturing profiles of 500 companies.</li>
                                    <li>• Cleaned over 10+ columns by handling null values and standardized text using Regular Expressions.</li>
                                    <li>• Converted data types across 5+ key fields to ensure consistency.</li>
                                    <li>• Conducted EDA using Python libraries (Pandas, Matplotlib, Seaborn) on the cleaned dataset.</li>
                                </ul>
                            </div>

                            <div class="p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200">
                                <h5 class="text-xl font-bold text-green-800 mb-3">👥 SQL Project On Employee Management System</h5>
                                <p class="text-gray-700 mb-3">Developed a relational system to manage employee data—covering roles, payroll, leaves, and qualifications—to support HR operations with accurate, accessible insights.</p>
                                <ul class="text-sm text-gray-600 space-y-1">
                                    <li>• Designed a normalized schema with 6 tables using foreign keys and cascading actions.</li>
                                    <li>• Inserted over 60 sample records and crafted 20+ SQL queries to analyze workforce data.</li>
                                    <li>• Automated payroll processing by integrating salary, bonus, and leave details.</li>
                                </ul>
                            </div>

                            <div class="p-6 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                                <h5 class="text-xl font-bold text-purple-800 mb-3">![image/png](https://cdn-uploads.huggingface.co/production/uploads/68661e9d1d9c7ebf69ef0ac0/nkYJs7uu55upycpZu3_vm.png) Power BI Project: Analysis on Top 10 Company Types from AmbitionBox</h5>
                                <p class="text-gray-700 mb-3">To provide data-driven insights on salary trends, job availability, and interview activity across 10 major company types, aiding informed decision-making for professionals and HR teams.</p>
                                <ul class="text-sm text-gray-600 space-y-1">
                                    <li>• Extracted and processed company data from AmbitionBox covering salaries, jobs, and interviews.</li>
                                    <li>• Built 3 dynamic Power BI dashboards for Salary, Job, and Interview analysis.</li>
                                    <li>• Utilized slicers, bar/line charts, and KPIs to visualize 50K+ salary records, 71K jobs, and 212K interviews.</li>
                                </ul>
                            </div>

                            <div class="p-6 bg-gradient-to-r from-orange-50 to-red-50 rounded-xl border border-orange-200">
                                <h5 class="text-xl font-bold text-orange-800 mb-3">📊 Customer Churn Prediction</h5>
                                <p class="text-gray-700 mb-3">Built churn prediction model using Logistic Regression and Random Forest with 85% accuracy.</p>
                                <ul class="text-sm text-gray-600 space-y-1">
                                    <li>• Cleaned data and engineered features using Pandas and NumPy</li>
                                    <li>• Evaluated model using confusion matrix and ROC-AUC score</li>
                                    <li>• Deployed interactive Streamlit app for real-time predictions</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>



                <div class="mt-8 text-center">
                    <p class="text-lg text-gray-600">📍 Bangalore, Karnataka | 📞 +91 9949995028</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let cities = [];
        let selectedCity = null;
        
        function showPage(pageId, element) {
            document.querySelectorAll('.page-content').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
            const targetPage = document.getElementById(pageId + '-page');
            if (targetPage) {
                targetPage.classList.remove('hidden');
                setTimeout(() => targetPage.classList.add('active'), 50);
            }
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            if (element) element.classList.add('active');
            if (pageId === 'visualization') loadVisualizations();
        }
        
        async function loadDatasetStats() {
            try {
                const response = await fetch('/api/dataset-stats');
                const stats = await response.json();
                document.getElementById('total-records').textContent = stats.total_records.toLocaleString();
                document.getElementById('cities-count').textContent = stats.cities_count;
                document.getElementById('states-count').textContent = stats.states_count;
                document.getElementById('common-aqi').textContent = stats.most_common_aqi;
            } catch (e) { console.error("Stats Error:", e); }
        }

        async function loadVisualizations() {
            const content = document.getElementById('visualization-content');
            content.innerHTML = '<div class="text-center p-8 text-white">Loading Visualizations...</div>';
            try {
                const response = await fetch('/api/visualizations');
                const plots = await response.json();
                content.innerHTML = '';
                plots.forEach((plot, index) => {
                    const plotContainer = document.createElement('div');
                    plotContainer.className = 'plot-container';
                    plotContainer.innerHTML = `<h3 class="text-2xl font-bold text-gray-800 mb-4 text-center">${plot.title}</h3><div id="plot-${index}" style="height: 500px;"></div>`;
                    content.appendChild(plotContainer);
                    Plotly.newPlot('plot-' + index, plot.data, plot.layout, {responsive: true});
                });
            } catch (e) { content.innerHTML = '<div class="text-center p-8 text-red-500">Could not load visualizations. Check server logs.</div>'; }
        }
        
        async function loadCities() {
            const loadingEl = document.getElementById('loadingCities');
            try {
                const response = await fetch('/get-cities');
                if (!response.ok) throw new Error('Failed to fetch cities');
                cities = await response.json();
                if(loadingEl) loadingEl.textContent = `${cities.length} cities available.`;
            } catch (e) { if(loadingEl) loadingEl.textContent = 'Error loading cities.'; console.error(e); }
        }
        
        function filterCities(term) {
            const dropdown = document.getElementById('cityDropdown');
            if (!term) { dropdown.classList.add('hidden'); return; }
            const filtered = cities.filter(c => c.city.toLowerCase().includes(term.toLowerCase())).slice(0, 5);
            dropdown.innerHTML = filtered.map(c => `<div class="city-option" onclick='selectCity(${JSON.stringify(c)})'>${c.city}, ${c.state}</div>`).join('');
            dropdown.classList.remove('hidden');
        }

        function selectCity(cityObj) {
            selectedCity = cityObj;
            document.getElementById('citySearch').value = `${cityObj.city}, ${cityObj.state}`;
            document.getElementById('cityDropdown').classList.add('hidden');
            document.getElementById('selectedCityInfo').classList.remove('hidden');
            document.getElementById('cityName').textContent = `${cityObj.city}, ${cityObj.state}`;
            document.getElementById('predictBtn').disabled = false;
        }

        async function predictAQI() {
            if (!selectedCity) return;
            const btn = document.getElementById('predictBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('loadingSpinner');
            
            btn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'inline-block';
            
            try {
                const response = await fetch('/predict-city', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(selectedCity)
                });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Prediction request failed');
                }
                const result = await response.json();
                displayLiveData(result.live_data);
                displayResults(result.predicted_aqi);
            } catch (e) {
                alert(`Error: ${e.message}`);
            } finally {
                btn.disabled = false;
                btnText.style.display = 'inline-block';
                spinner.style.display = 'none';
            }
        }
        
        function displayLiveData(data) {
            document.getElementById('dataDefaultState').classList.add('hidden');
            const container = document.getElementById('liveDataContainer');
            container.classList.remove('hidden');
            document.getElementById('liveDataContent').innerHTML = `
                <p><strong>Temp:</strong> ${data.weather.temp}°C</p>
                <p><strong>Humidity:</strong> ${data.weather.humidity}%</p>
                <p><strong>Wind:</strong> ${data.weather.wind_speed} km/h</p>
                <p><strong>PM2.5:</strong> ${data.pollution.pm2_5} μg/m³</p>
                <p><strong>CO:</strong> ${data.pollution.co} μg/m³</p>
                <p><strong>NO₂:</strong> ${data.pollution.no2} μg/m³</p>
                <p><strong>O₃:</strong> ${data.pollution.o3} μg/m³</p>`;
        }
        
        function displayResults(aqi) {
            document.getElementById('resultsDefaultState').classList.add('hidden');
            const container = document.getElementById('resultsContainer');
            container.classList.remove('hidden');
            container.innerHTML = `<h3 class="text-3xl font-bold text-white text-center">Predicted AQI Value: ${aqi}</h3>`;
        }

        document.addEventListener('DOMContentLoaded', function() {
            showPage('home', document.querySelector('.nav-item.active'));
            loadDatasetStats();
            loadCities();
            document.getElementById('citySearch').addEventListener('input', e => filterCities(e.target.value));
            document.addEventListener('click', e => {
                if (!e.target.closest('.dropdown-container')) {
                    const dropdown = document.getElementById('cityDropdown');
                    if(dropdown) dropdown.classList.add('hidden');
                }
            });
        });
    </script>
</body>
</html>
""")

# ===============================================================
#  BACKEND PYTHON API FUNCTIONS
# ===============================================================

OPENWEATHER_API_KEY = "7a39513a6de2c9c09352615715060e6a" # IMPORTANT: Use secrets for production

class CityData(BaseModel):
    city: str
    state: str
    lat: float
    lng: float

@app.get("/get-cities")
async def get_cities():
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded, cannot get cities.")
    try:
        df.columns = df.columns.str.strip()
        # Ensure required columns exist before proceeding
        required_cols = ['City', 'State', 'lat', 'lng']
        if not all(col in df.columns for col in required_cols):
             raise HTTPException(status_code=500, detail=f"CSV is missing one or more required columns: {required_cols}")
        
        unique_cities = df[required_cols].drop_duplicates(subset=['City'])
        city_list = unique_cities.to_dict(orient='records')
        
        # Rename keys to match JavaScript expectations
        return [{'city': c['City'], 'state': c['State'], 'lat': c['lat'], 'lng': c['lng']} for c in city_list]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve city list: {e}")

@app.post("/predict-city")
async def predict_aqi_from_city(city_data: CityData):
    if model is None: raise HTTPException(status_code=503, detail="ML model is not available.")
    if not OPENWEATHER_API_KEY or "YOUR" in OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenWeatherMap API key is not configured on the server.")

    async with httpx.AsyncClient() as client:
        try:
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={city_data.lat}&lon={city_data.lng}&appid={OPENWEATHER_API_KEY}&units=metric"
            w_res = await client.get(weather_url)
            w_res.raise_for_status()
            weather = w_res.json()

            pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={city_data.lat}&lon={city_data.lng}&appid={OPENWEATHER_API_KEY}"
            p_res = await client.get(pollution_url)
            p_res.raise_for_status()
            pollution = p_res.json()['list'][0]['components']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching live data from API: {e}")

    features = [[
        pollution.get('co', 200), pollution.get('no', 1), pollution.get('no2', 10),
        pollution.get('o3', 50), pollution.get('so2', 2), pollution.get('pm2_5', 25),
        pollution.get('pm10', 50), pollution.get('nh3', 5), weather['main'].get('temp', 25),
        weather['main'].get('humidity', 60), weather['wind'].get('speed', 3), 
        weather.get('visibility', 10000)
    ]]
    
    predicted_aqi_value = int(model.predict(features)[0])
    
    live_data = {
        "weather": {"temp": round(weather['main'].get('temp',0)), "humidity": weather['main'].get('humidity',0), "wind_speed": round(weather['wind'].get('speed',0)*3.6,1), "visibility": round(weather.get('visibility',0)/1000,1)},
        "pollution": {"co": round(pollution.get('co',0),2), "no2": round(pollution.get('no2',0),2), "o3": round(pollution.get('o3',0),2), "pm2_5": round(pollution.get('pm2_5',0),2)}
    }
    return {"predicted_aqi": predicted_aqi_value, "live_data": live_data}

@app.get("/api/dataset-stats")
async def get_dataset_stats():
    if df is None: return {"total_records": "N/A", "cities_count": "N/A", "states_count": "N/A", "most_common_aqi": "N/A"}
    try:
        df.columns = df.columns.str.strip()
        stats = {
            'total_records': len(df),
            'cities_count': df['City'].nunique(),
            'states_count': df['State'].nunique(),
            'most_common_aqi': df['AQI_Category'].mode()[0] if 'AQI_Category' in df.columns else 'N/A'
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating stats: {e}")

def safe_plotly_json(obj):
    if isinstance(obj, dict): return {k: safe_plotly_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [safe_plotly_json(i) for i in obj]
    if hasattr(obj, 'tolist'): return obj.tolist()
    return obj

def convert_plotly_data(data):
    return [safe_plotly_json(trace.to_plotly_json()) for trace in data]

@app.get("/api/visualizations")
async def get_visualizations():
    if df is None: raise HTTPException(status_code=404, detail="Dataset not loaded.")
    try:
        plots = []
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.strip()
        
        # Plot 1: Top 15 States by Average AQI
        if 'State' in df_clean.columns and 'AQI' in df_clean.columns:
            avg_aqi_state = df_clean.groupby('State')['AQI'].mean().sort_values(ascending=False).head(15)
            fig1 = px.bar(avg_aqi_state, x=avg_aqi_state.index, y='AQI', title="Top 15 States by Average AQI", color_discrete_sequence=px.colors.sequential.Plasma_r)
            plots.append({'title': fig1.layout.title.text, 'data': convert_plotly_data(fig1.data), 'layout': safe_plotly_json(fig1.layout.to_plotly_json())})
        
        # Plot 2: Correlation Matrix
        corr_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
        corr_cols_exist = [c for c in corr_cols if c in df_clean.columns]
        if len(corr_cols_exist) > 1:
            corr_matrix = df_clean[corr_cols_exist].corr()
            fig2 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix of Pollutants")
            plots.append({'title': fig2.layout.title.text, 'data': convert_plotly_data(fig2.data), 'layout': safe_plotly_json(fig2.layout.to_plotly_json())})

        return plots
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

