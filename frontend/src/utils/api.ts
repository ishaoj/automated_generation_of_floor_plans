import axios from 'axios';
import { FloorPlanRequest, FloorPlanResponse } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function generateFloorPlans(request: FloorPlanRequest): Promise<FloorPlanResponse> {
  try {
    const response = await api.post<FloorPlanResponse>('/api/generate', request);
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to generate floor plans. Please try again.');
  }
}

export async function getVastuRules() {
  const response = await api.get('/api/vastu-rules');
  return response.data;
}

export async function getRoomTypes() {
  const response = await api.get('/api/room-types');
  return response.data;
}

export async function checkHealth() {
  const response = await api.get('/health');
  return response.data;
}

export interface PreviewSample {
  semantic_map: string;
  generated:    string;
  target:       string;
}

export interface PreviewResponse {
  epoch:          number;
  best_val_loss:  number;
  checkpoint:     string;
  num_samples:    number;
  samples:        PreviewSample[];
}

export async function getPreview(numSamples: number = 4): Promise<PreviewResponse> {
  try {
    const response = await api.get<PreviewResponse>('/api/preview', {
      params: { num_samples: numSamples },
      timeout: 60000,
    });
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to load preview. Is the backend running?');
  }
}
