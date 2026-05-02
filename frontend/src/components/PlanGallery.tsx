import { useState, useEffect } from 'react';
import { GeneratedPlan } from '@/utils/types';
import VastuScoreCard from './VastuScoreCard';
import FloorPlanViewer from './FloorPlanViewer';

interface PlanGalleryProps {
  plans: GeneratedPlan[];
  requestSummary?: {
    plot_size: string;
    facing: string;
    bedrooms: number;
    bathrooms: number;
  } | null;
}

export default function PlanGallery({ plans, requestSummary }: PlanGalleryProps) {
  const [selectedPlan, setSelectedPlan] = useState<GeneratedPlan | null>(
    plans.length > 0 ? plans[0] : null
  );
  const [isAnimating, setIsAnimating] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);

  // Show confetti on first load if score is high
  useEffect(() => {
    const bestScore = Math.max(...plans.map(p => p.vastu_score.total));
    if (bestScore >= 90) {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 2000);
    }
  }, [plans]);

  const handlePlanSelect = (plan: GeneratedPlan) => {
    if (plan.plan_id !== selectedPlan?.plan_id) {
      setIsAnimating(true);
      setSelectedPlan(plan);
      setTimeout(() => setIsAnimating(false), 300);
    }
  };

  const downloadPlan = (plan: GeneratedPlan) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${plan.image_base64}`;
    link.download = `floor-plan-${plan.plan_id}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const totalArea = selectedPlan
    ? selectedPlan.layout.rooms.reduce((sum, room) => sum + room.area, 0)
    : 0;
  const plotArea = selectedPlan
    ? selectedPlan.layout.plot_width * selectedPlan.layout.plot_length
    : 0;
  const coverage = plotArea > 0 ? Math.round((totalArea / plotArea) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Header with Summary */}
      <div className={`bg-white rounded-2xl shadow-lg border border-gray-100 p-5 animate-fade-in-down ${showConfetti ? 'confetti-burst' : ''}`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
              <svg className="w-6 h-6 text-vastu-secondary animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="animate-fade-in">{plans.length} Floor Plan{plans.length > 1 ? 's' : ''} Generated</span>
            </h3>
            {requestSummary && (
              <p className="text-sm text-gray-500 mt-1 animate-fade-in" style={{ animationDelay: '200ms' }}>
                {requestSummary.plot_size} | {requestSummary.facing} facing | {requestSummary.bedrooms} BR, {requestSummary.bathrooms} Bath
              </p>
            )}
          </div>

          {/* Quick Stats with animation */}
          <div className="flex gap-3">
            <div className="px-4 py-2 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-100 hover:shadow-md transition-all duration-300 hover:scale-105 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
              <p className="text-xs text-green-600 font-medium">Best Score</p>
              <p className="text-lg font-bold text-green-700 counter">
                {Math.max(...plans.map(p => p.vastu_score.total))}
              </p>
            </div>
            <div className="px-4 py-2 bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl border border-orange-100 hover:shadow-md transition-all duration-300 hover:scale-105 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
              <p className="text-xs text-orange-600 font-medium">Coverage</p>
              <p className="text-lg font-bold text-orange-700 counter">{coverage}%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Plan Thumbnails */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-5 animate-fade-in-up" style={{ animationDelay: '150ms' }}>
        <h4 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
          Select a Design
        </h4>
        <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
          {plans.map((plan, index) => (
            <button
              key={plan.plan_id}
              onClick={() => handlePlanSelect(plan)}
              className={`flex-shrink-0 relative rounded-xl overflow-hidden transition-all duration-300 transform hover:scale-105 active:scale-100 ${
                selectedPlan?.plan_id === plan.plan_id
                  ? 'ring-4 ring-vastu-secondary shadow-xl shadow-orange-200 scale-105'
                  : 'ring-1 ring-gray-200 hover:ring-gray-300 hover:shadow-lg'
              }`}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <img
                src={`data:image/png;base64,${plan.image_base64}`}
                alt={`Floor Plan ${index + 1}`}
                className={`w-36 h-36 object-cover transition-all duration-300 ${
                  selectedPlan?.plan_id === plan.plan_id ? 'brightness-100' : 'brightness-95 hover:brightness-100'
                }`}
              />
              {/* Overlay gradient with hover effect */}
              <div className={`absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent transition-opacity duration-300 ${
                selectedPlan?.plan_id === plan.plan_id ? 'opacity-100' : 'opacity-80 hover:opacity-100'
              }`} />

              {/* Option badge with animation */}
              <div className="absolute bottom-2 left-2 right-2 flex justify-between items-end">
                <span className="text-white text-sm font-semibold drop-shadow-lg">
                  Option {index + 1}
                </span>
              </div>

              {/* Score badge with pulse animation for high scores */}
              <div
                className={`absolute top-2 right-2 px-2.5 py-1 rounded-lg text-xs font-bold text-white shadow-lg transition-all duration-300 ${
                  plan.vastu_score.total >= 80
                    ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                    : plan.vastu_score.total >= 60
                    ? 'bg-gradient-to-r from-yellow-500 to-amber-500'
                    : 'bg-gradient-to-r from-red-500 to-red-600'
                } ${plan.vastu_score.total >= 90 ? 'animate-pulse' : ''}`}
              >
                {plan.vastu_score.total}
              </div>

              {/* Selected indicator with bounce animation */}
              {selectedPlan?.plan_id === plan.plan_id && (
                <div className="absolute top-2 left-2 animate-bounce-in">
                  <div className="w-7 h-7 bg-vastu-secondary rounded-full flex items-center justify-center shadow-lg">
                    <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                </div>
              )}

              {/* Hover overlay */}
              <div className={`absolute inset-0 bg-vastu-secondary/10 opacity-0 hover:opacity-100 transition-opacity duration-300 ${
                selectedPlan?.plan_id === plan.plan_id ? 'hidden' : ''
              }`} />
            </button>
          ))}
        </div>
      </div>

      {/* Selected Plan Detail with transition */}
      {selectedPlan && (
        <div className={`grid lg:grid-cols-3 gap-6 transition-all duration-300 ${isAnimating ? 'opacity-50 scale-[0.99]' : 'opacity-100 scale-100'}`}>
          {/* Floor Plan Image */}
          <div className="lg:col-span-2 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
            <FloorPlanViewer plan={selectedPlan} onDownload={() => downloadPlan(selectedPlan)} />
          </div>

          {/* Vastu Score Details */}
          <div className="lg:col-span-1 space-y-4">
            <div className="animate-fade-in-up" style={{ animationDelay: '250ms' }}>
              <VastuScoreCard score={selectedPlan.vastu_score} />
            </div>

            {/* Design Improvement Suggestions */}
            {selectedPlan.design_suggestions && selectedPlan.design_suggestions.length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-5 animate-fade-in-up" style={{ animationDelay: '280ms' }}>
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                  Design Improvements
                  <span className="ml-auto text-xs text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full font-medium">
                    {selectedPlan.design_suggestions.length}
                  </span>
                </h4>
                <ul className="space-y-2">
                  {selectedPlan.design_suggestions.map((suggestion, index) => (
                    <li
                      key={index}
                      className="flex items-start gap-2.5 text-sm text-gray-600 p-2.5 bg-blue-50 rounded-lg border border-blue-100 hover:bg-blue-100 hover:border-blue-200 transition-all duration-200 cursor-default"
                    >
                      <span className="flex-shrink-0 mt-0.5 w-5 h-5 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold">
                        {index + 1}
                      </span>
                      <span className="leading-relaxed">{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Room List with staggered animation */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-5 animate-fade-in-up card-hover" style={{ animationDelay: '300ms' }}>
              <h4 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                </svg>
                Room Details
              </h4>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-2">
                {selectedPlan.layout.rooms.map((room, index) => (
                  <div
                    key={room.id}
                    className="flex justify-between items-center p-2.5 bg-gray-50 rounded-lg hover:bg-gray-100 hover:scale-[1.02] transition-all duration-200 cursor-default"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${getRoomColor(room.room_type)} transition-transform hover:scale-125`}></div>
                      <span className="text-sm text-gray-700 font-medium">
                        {room.room_type.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded shadow-sm">
                      {room.width.toFixed(0)}' × {room.height.toFixed(0)}' ({room.area.toFixed(0)} ft²)
                    </span>
                  </div>
                ))}
              </div>

              {/* Total summary with animated counter */}
              <div className="mt-4 pt-4 border-t border-gray-100 flex justify-between items-center">
                <span className="text-sm font-medium text-gray-600">Total Coverage</span>
                <div className="flex items-center gap-2">
                  <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-vastu-secondary to-orange-400 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${coverage}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-bold text-vastu-secondary counter">
                    {totalArea.toFixed(0)} ft² ({coverage}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getRoomColor(roomType: string): string {
  const colors: Record<string, string> = {
    master_bedroom: 'bg-amber-400',
    bedroom: 'bg-amber-300',
    living_room: 'bg-blue-400',
    kitchen: 'bg-red-400',
    bathroom: 'bg-purple-400',
    dining: 'bg-green-400',
    puja_room: 'bg-yellow-400',
    parking: 'bg-slate-400',
    balcony: 'bg-teal-400',
    staircase: 'bg-gray-400',
    ots: 'bg-sky-400',
  };
  return colors[roomType] || 'bg-gray-300';
}
