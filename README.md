# Vastu-AI

AI-Powered Vastu-Compliant Floor Plan Generator

## Overview

Vastu-AI is a web application that generates floor plans based on user requirements while adhering to Vastu Shastra principles. It uses a Graph Neural Network (GNN) combined with constraint solving to create multiple optimized floor plan variations.

## Features

- **Vastu-Compliant Generation**: Automatically places rooms according to Vastu Shastra guidelines
- **Multiple Variations**: Generates 3-5 different floor plan options per request
- **Compliance Scoring**: Each plan receives a Vastu compliance score (0-100)
- **Customizable Input**: Specify plot size, facing direction, number of rooms, and optional spaces
- **Visual Output**: Clean 2D floor plan images with room labels and dimensions

## Tech Stack

### Backend
- Python 3.10+
- FastAPI
- OR-Tools (Constraint Solver)
- PyTorch + PyTorch Geometric (GNN)
- Pillow (Image Rendering)

### Frontend
- Next.js 14
- React 18
- TailwindCSS
- React Query

## Project Structure

```
vastu-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # API endpoints
│   │   ├── core/             # Vastu rules, constants
│   │   ├── models/           # Pydantic schemas
│   │   └── services/         # Core business logic
│   ├── ml/                   # ML models (GNN)
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Next.js pages
│   │   ├── hooks/            # Custom hooks
│   │   └── utils/            # API client, types
│   └── package.json
│
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate` | Generate floor plans |
| GET | `/api/vastu-rules` | Get Vastu rules |
| GET | `/api/room-types` | Get room configurations |
| GET | `/health` | Health check |

### Example Request

```json
POST /api/generate
{
  "plot_length": 40,
  "plot_width": 30,
  "facing_direction": "north",
  "num_bedrooms": 2,
  "num_bathrooms": 2,
  "include_kitchen": true,
  "include_living_room": true,
  "include_dining": true,
  "include_puja_room": false,
  "num_variations": 3
}
```

## Vastu Rules

The system encodes key Vastu Shastra principles:

| Room Type | Ideal Direction |
|-----------|-----------------|
| Master Bedroom | Southwest |
| Kitchen | Southeast |
| Living Room | North/East |
| Puja Room | Northeast |
| Bathroom | Northwest |
| Entrance | North/East |

## Development Phases

1. **Phase 1** (Current): Constraint-based generation
2. **Phase 2**: GNN integration for smarter layouts
3. **Phase 3**: Diffusion model for visual refinement

## License

MIT License
