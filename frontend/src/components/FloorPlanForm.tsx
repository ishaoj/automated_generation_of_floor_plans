import { useState } from 'react';
import { FloorPlanRequest } from '@/utils/types';

interface FloorPlanFormProps {
  onSubmit: (request: FloorPlanRequest) => void;
  isLoading: boolean;
}

export default function FloorPlanForm({ onSubmit, isLoading }: FloorPlanFormProps) {
  const [formData, setFormData] = useState<FloorPlanRequest>({
    plot_length: 40,
    plot_width: 30,
    facing_direction: 'north',
    num_bedrooms: 2,
    num_bathrooms: 2,
    include_kitchen: true,
    include_living_room: true,
    include_dining: true,
    include_puja_room: false,
    include_parking: false,
    include_balcony: false,
    include_staircase: false,
    include_ots: false,
    num_variations: 3 as const,
    plot_type: 'open_all',
    corner_open_side: 'right',
  });

  const [activeSection, setActiveSection] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;

    if (type === 'checkbox') {
      const checked = (e.target as HTMLInputElement).checked;
      setFormData((prev) => ({ ...prev, [name]: checked }));
    } else if (type === 'number') {
      setFormData((prev) => ({ ...prev, [name]: parseFloat(value) || 0 }));
    } else {
      setFormData((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const plotArea = formData.plot_length * formData.plot_width;

  const directionIcons: Record<string, JSX.Element> = {
    north: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />,
    south: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />,
    east: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />,
    west: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />,
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Plot Dimensions */}
      <div
        className={`space-y-3 transition-all duration-300 ${activeSection === 'plot' ? 'scale-[1.02]' : ''}`}
        onFocus={() => setActiveSection('plot')}
        onBlur={() => setActiveSection(null)}
      >
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <svg className="w-4 h-4 text-vastu-secondary animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
          Plot Dimensions
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="group">
            <label className="block text-xs text-gray-500 mb-1.5 group-focus-within:text-vastu-secondary transition-colors">Length (ft)</label>
            <input
              type="number"
              name="plot_length"
              value={formData.plot_length}
              onChange={handleChange}
              min={15}
              max={200}
              className="w-full px-3 py-2.5 border border-gray-200 rounded-xl focus:ring-2 focus:ring-vastu-secondary/20 focus:border-vastu-secondary transition-all duration-300 text-gray-800 hover:border-gray-300 focus:scale-[1.02] input-focus"
            />
          </div>
          <div className="group">
            <label className="block text-xs text-gray-500 mb-1.5 group-focus-within:text-vastu-secondary transition-colors">Width (ft)</label>
            <input
              type="number"
              name="plot_width"
              value={formData.plot_width}
              onChange={handleChange}
              min={15}
              max={200}
              className="w-full px-3 py-2.5 border border-gray-200 rounded-xl focus:ring-2 focus:ring-vastu-secondary/20 focus:border-vastu-secondary transition-all duration-300 text-gray-800 hover:border-gray-300 focus:scale-[1.02] input-focus"
            />
          </div>
        </div>
        <div className="flex items-center justify-between px-3 py-2.5 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-100 hover:shadow-md transition-all duration-300">
          <span className="text-xs text-gray-600">Total Area</span>
          <span className="text-sm font-bold text-vastu-secondary counter">{plotArea.toLocaleString()} sq ft</span>
        </div>
      </div>

      {/* Facing Direction */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <svg className="w-4 h-4 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
          </svg>
          Main Entrance Direction
        </div>
        <div className="grid grid-cols-4 gap-2">
          {['north', 'south', 'east', 'west'].map((dir, i) => (
            <button
              key={dir}
              type="button"
              onClick={() => setFormData((prev) => ({ ...prev, facing_direction: dir as any }))}
              className={`flex flex-col items-center gap-1 py-2.5 px-2 rounded-xl text-xs font-medium transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                formData.facing_direction === dir
                  ? 'bg-vastu-secondary text-white shadow-lg shadow-orange-200 scale-105'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200 hover:border-gray-300'
              }`}
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <svg className={`w-4 h-4 transition-transform duration-300 ${formData.facing_direction === dir ? 'animate-bounce' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {directionIcons[dir]}
              </svg>
              {dir.charAt(0).toUpperCase() + dir.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Plot Type */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <svg className="w-4 h-4 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
          Plot Configuration
        </div>
        <div className="space-y-2">
          {[
            { value: 'open_all', label: 'Standalone', desc: 'All 4 sides open', icon: '🏠' },
            { value: 'open_one', label: 'Row House', desc: 'Only front open', icon: '🏘️' },
            { value: 'open_two_corner', label: 'Corner Plot', desc: 'Front + side open', icon: '🏗️' },
          ].map(({ value, label, desc, icon }, i) => (
            <button
              key={value}
              type="button"
              onClick={() => setFormData((prev) => ({ ...prev, plot_type: value as any }))}
              className={`w-full flex items-center gap-3 text-left py-3 px-4 rounded-xl transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] ${
                formData.plot_type === value
                  ? 'bg-gradient-to-r from-vastu-secondary to-orange-500 text-white shadow-lg shadow-orange-200'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200 hover:border-gray-300'
              }`}
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <span className={`text-xl transition-transform duration-300 ${formData.plot_type === value ? 'animate-bounce' : 'hover:scale-110'}`}>{icon}</span>
              <div className="flex-1">
                <span className="font-medium text-sm">{label}</span>
                <span className={`text-xs block transition-colors ${formData.plot_type === value ? 'text-orange-100' : 'text-gray-400'}`}>
                  {desc}
                </span>
              </div>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center transition-all duration-300 ${
                formData.plot_type === value
                  ? 'bg-white/20 scale-100'
                  : 'bg-gray-200 scale-0'
              }`}>
                {formData.plot_type === value && (
                  <svg className="w-4 h-4 text-white animate-scale-in" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
            </button>
          ))}
        </div>

        {/* Corner side selector with animation */}
        <div className={`overflow-hidden transition-all duration-500 ease-out ${formData.plot_type === 'open_two_corner' ? 'max-h-40 opacity-100' : 'max-h-0 opacity-0'}`}>
          <div className="mt-2 p-3 bg-orange-50 rounded-xl border border-orange-100">
            <label className="block text-xs font-medium text-gray-600 mb-2">
              Open side (from entrance view)
            </label>
            <div className="grid grid-cols-2 gap-2">
              {['left', 'right'].map((side) => (
                <button
                  key={side}
                  type="button"
                  onClick={() => setFormData((prev) => ({ ...prev, corner_open_side: side as any }))}
                  className={`py-2.5 px-3 rounded-lg text-sm font-medium transition-all duration-300 transform hover:scale-105 active:scale-95 ${
                    formData.corner_open_side === side
                      ? 'bg-vastu-secondary text-white shadow-md'
                      : 'bg-white text-gray-600 hover:bg-gray-50 border border-gray-200'
                  }`}
                >
                  {side.charAt(0).toUpperCase() + side.slice(1)} Side
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Room Counts */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <svg className="w-4 h-4 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
          Room Count
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="group">
            <label className="block text-xs text-gray-500 mb-1.5">Bedrooms</label>
            <select
              name="num_bedrooms"
              value={formData.num_bedrooms}
              onChange={handleChange}
              className="w-full px-3 py-2.5 border border-gray-200 rounded-xl focus:ring-2 focus:ring-vastu-secondary/20 focus:border-vastu-secondary transition-all duration-300 text-gray-800 bg-white hover:border-gray-300 cursor-pointer"
            >
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <option key={n} value={n}>
                  {n} {n === 1 ? 'Bedroom' : 'Bedrooms'}
                </option>
              ))}
            </select>
          </div>
          <div className="group">
            <label className="block text-xs text-gray-500 mb-1.5">Bathrooms</label>
            <select
              name="num_bathrooms"
              value={formData.num_bathrooms}
              onChange={handleChange}
              className="w-full px-3 py-2.5 border border-gray-200 rounded-xl focus:ring-2 focus:ring-vastu-secondary/20 focus:border-vastu-secondary transition-all duration-300 text-gray-800 bg-white hover:border-gray-300 cursor-pointer"
            >
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n} {n === 1 ? 'Bathroom' : 'Bathrooms'}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Optional Rooms */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <svg className="w-4 h-4 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          Additional Rooms
        </div>
        <div className="grid grid-cols-2 gap-2">
          {[
            { key: 'include_kitchen', label: 'Kitchen', icon: '🍳' },
            { key: 'include_living_room', label: 'Living Room', icon: '🛋️' },
            { key: 'include_dining', label: 'Dining', icon: '🍽️' },
            { key: 'include_puja_room', label: 'Puja Room', icon: '🪔' },
            { key: 'include_parking', label: 'Parking', icon: '🚗' },
            { key: 'include_balcony', label: 'Balcony', icon: '🌿' },
            { key: 'include_staircase', label: 'Staircase', icon: '🪜' },
            { key: 'include_ots', label: 'Open to Sky', icon: '☀️' },
          ].map(({ key, label, icon }, i) => {
            const isChecked = (formData as any)[key];
            return (
              <label
                key={key}
                className={`flex items-center gap-2.5 p-2.5 rounded-xl cursor-pointer transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] ${
                  isChecked
                    ? 'bg-gradient-to-r from-orange-50 to-amber-50 border border-orange-200 shadow-sm'
                    : 'bg-gray-50 border border-transparent hover:bg-gray-100 hover:border-gray-200'
                }`}
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <input
                  type="checkbox"
                  name={key}
                  checked={isChecked}
                  onChange={handleChange}
                  className="sr-only"
                />
                <div className={`w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all duration-300 ${
                  isChecked
                    ? 'bg-vastu-secondary border-vastu-secondary scale-110'
                    : 'border-gray-300 bg-white hover:border-gray-400'
                }`}>
                  <svg
                    className={`w-3 h-3 text-white transition-all duration-300 ${isChecked ? 'scale-100 opacity-100' : 'scale-0 opacity-0'}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <span className={`text-base transition-transform duration-300 ${isChecked ? 'scale-110' : ''}`}>{icon}</span>
                <span className={`text-sm transition-all duration-300 ${isChecked ? 'text-gray-800 font-medium' : 'text-gray-600'}`}>{label}</span>
              </label>
            );
          })}
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isLoading}
        className={`w-full py-4 px-4 rounded-xl font-semibold text-white transition-all duration-300 flex items-center justify-center gap-2 transform hover:scale-[1.02] active:scale-[0.98] ripple btn-press ${
          isLoading
            ? 'bg-gray-400 cursor-not-allowed scale-100'
            : 'bg-gradient-to-r from-vastu-secondary to-orange-600 hover:from-orange-600 hover:to-orange-700 shadow-lg shadow-orange-200 hover:shadow-xl hover:shadow-orange-300'
        }`}
      >
        {isLoading ? (
          <>
            <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="animate-pulse">Generating...</span>
          </>
        ) : (
          <>
            <svg className="w-5 h-5 transition-transform group-hover:rotate-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Generate Floor Plans
          </>
        )}
      </button>
    </form>
  );
}
