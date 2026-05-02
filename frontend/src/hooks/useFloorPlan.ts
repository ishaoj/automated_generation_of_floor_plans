import { useMutation, useQuery } from '@tanstack/react-query';
import { generateFloorPlans, getVastuRules, getRoomTypes } from '@/utils/api';
import { FloorPlanRequest } from '@/utils/types';

export function useGenerateFloorPlans() {
  return useMutation({
    mutationFn: (request: FloorPlanRequest) => generateFloorPlans(request),
  });
}

export function useVastuRules() {
  return useQuery({
    queryKey: ['vastu-rules'],
    queryFn: getVastuRules,
    staleTime: Infinity, // Rules don't change
  });
}

export function useRoomTypes() {
  return useQuery({
    queryKey: ['room-types'],
    queryFn: getRoomTypes,
    staleTime: Infinity,
  });
}
