export interface FloorPlanRequest {
  plot_length: number;
  plot_width: number;
  facing_direction: 'north' | 'south' | 'east' | 'west';
  num_bedrooms: number;
  num_bathrooms: number;
  include_kitchen: boolean;
  include_living_room: boolean;
  include_dining: boolean;
  include_puja_room: boolean;
  include_parking: boolean;
  include_balcony: boolean;
  include_staircase: boolean;
  include_ots: boolean;  // Open to Sky courtyard
  num_variations: number;

  // Plot type configuration
  plot_type: 'open_all' | 'open_one' | 'open_two_corner';
  corner_open_side?: 'left' | 'right';  // Only needed for corner plots
}

export interface Room {
  id: string;
  room_type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  area: number;
  zone: string;
}

export interface Layout {
  rooms: Room[];
  plot_length: number;
  plot_width: number;
  facing_direction: string;
  entrance_position: [number, number];
}

export interface VastuScore {
  total: number;
  room_placement: number;
  entrance: number;
  center_clear: number;
  proportions: number;
  suggestions: string[];
}

export interface GeneratedPlan {
  plan_id: string;
  layout: Layout;
  vastu_score: VastuScore;
  image_base64: string;
  image_format: string;
  design_suggestions: string[];
}

export interface FloorPlanResponse {
  success: boolean;
  message: string;
  plans: GeneratedPlan[];
  request_summary: {
    plot_size: string;
    facing: string;
    bedrooms: number;
    bathrooms: number;
  };
}
