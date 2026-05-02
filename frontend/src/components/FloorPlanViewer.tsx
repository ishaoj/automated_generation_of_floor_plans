import { GeneratedPlan } from '@/utils/types';

interface FloorPlanViewerProps {
  plan: GeneratedPlan;
  onDownload: () => void;
}

const ROOM_COLORS: Record<string, { bg: string; label: string }> = {
  master_bedroom: { bg: '#FFC864', label: 'Master Bedroom' },
  bedroom:        { bg: '#FFC864', label: 'Bedroom' },
  living_room:    { bg: '#50DC78', label: 'Living Room' },
  kitchen:        { bg: '#FFFF50', label: 'Kitchen' },
  bathroom:       { bg: '#64B4FF', label: 'Bathroom' },
  dining:         { bg: '#A0DCA0', label: 'Dining' },
  puja_room:      { bg: '#DC64DC', label: 'Puja Room' },
  parking:        { bg: '#B4B4B4', label: 'Parking' },
  balcony:        { bg: '#A0DCF0', label: 'Balcony' },
  staircase:      { bg: '#FF8C8C', label: 'Staircase' },
  store:          { bg: '#B4966E', label: 'Store' },
  ots:            { bg: '#C8FFC8', label: 'Open to Sky' },
  corridor:       { bg: '#F0DCC0', label: 'Corridor' },
  utility:        { bg: '#C8BEA0', label: 'Utility' },
  entrance:       { bg: '#FFF0C8', label: 'Entrance' },
};

export default function FloorPlanViewer({ plan, onDownload }: FloorPlanViewerProps) {
  const roomTypes = [...new Set(plan.layout.rooms.map(r => r.room_type))];

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b bg-gradient-to-r from-gray-50 to-slate-50 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-vastu-primary/10 rounded-xl flex items-center justify-center">
            <svg className="w-5 h-5 text-vastu-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <div>
            <h3 className="font-bold text-gray-800">Floor Plan</h3>
            <div className="flex items-center gap-3 text-sm text-gray-500 mt-0.5">
              <span className="flex items-center gap-1">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
                {plan.layout.plot_width}' × {plan.layout.plot_length}'
              </span>
              <span className="text-gray-300">|</span>
              <span className="flex items-center gap-1 capitalize">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                </svg>
                {plan.layout.facing_direction} facing
              </span>
            </div>
          </div>
        </div>
        <button
          onClick={onDownload}
          className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-vastu-primary to-slate-700 text-white rounded-xl hover:from-slate-700 hover:to-slate-800 transition-all shadow-md hover:shadow-lg text-sm font-medium"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download PNG
        </button>
      </div>


      {/* Image */}
      <div className="p-5 bg-gray-50">
        <div className="relative bg-white rounded-xl shadow-inner border border-gray-200 p-3 overflow-hidden">
          <img
            src={`data:image/png;base64,${plan.image_base64}`}
            alt="Floor Plan"
            className="w-full h-auto"
          />
        </div>
      </div>

      {/* Legend */}
      <div className="px-5 py-4 border-t bg-white">
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
          </svg>
          <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Room Legend</span>
        </div>
        <div className="flex flex-wrap gap-3">
          {roomTypes.map((roomType) => {
            const color = ROOM_COLORS[roomType] || { bg: '#E5E7EB', label: roomType.replace(/_/g, ' ') };
            return (
              <div key={roomType} className="flex items-center gap-2 px-2.5 py-1.5 bg-gray-50 rounded-lg">
                <div
                  className="w-4 h-4 rounded shadow-sm border border-gray-200"
                  style={{ backgroundColor: color.bg }}
                />
                <span className="text-xs text-gray-600 font-medium capitalize">{color.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
