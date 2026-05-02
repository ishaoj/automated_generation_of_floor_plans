import { useState, useEffect } from 'react';
import { VastuScore } from '@/utils/types';

interface VastuScoreCardProps {
  score: VastuScore;
}

export default function VastuScoreCard({ score }: VastuScoreCardProps) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const [showBreakdown, setShowBreakdown] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Animate score counter on mount
  useEffect(() => {
    const duration = 1500;
    const steps = 60;
    const increment = score.total / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= score.total) {
        setAnimatedScore(score.total);
        clearInterval(timer);
      } else {
        setAnimatedScore(Math.floor(current));
      }
    }, duration / steps);

    // Trigger staggered animations
    setTimeout(() => setShowBreakdown(true), 500);
    setTimeout(() => setShowSuggestions(true), 1000);

    return () => clearInterval(timer);
  }, [score.total]);

  const getScoreColor = (value: number) => {
    if (value >= 80) return 'text-green-600';
    if (value >= 60) return 'text-amber-600';
    return 'text-red-600';
  };

  const getScoreBg = (value: number) => {
    if (value >= 80) return 'from-green-500 to-emerald-600';
    if (value >= 60) return 'from-amber-500 to-orange-500';
    return 'from-red-500 to-rose-600';
  };

  const getScoreRingColor = (value: number) => {
    if (value >= 80) return 'stroke-green-500';
    if (value >= 60) return 'stroke-amber-500';
    return 'stroke-red-500';
  };

  const scoreItems = [
    { label: 'Room Placement', value: score.room_placement, icon: '🏠' },
    { label: 'Entrance', value: score.entrance, icon: '🚪' },
    { label: 'Center (Brahmasthan)', value: score.center_clear, icon: '☯️' },
    { label: 'Proportions', value: score.proportions, icon: '📐' },
  ];

  // Calculate circle progress
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="px-5 py-4 bg-gradient-to-r from-gray-50 to-orange-50 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 bg-vastu-secondary/10 rounded-xl flex items-center justify-center float-icon ${score.total >= 80 ? 'animate-bounce' : ''}`}>
            <svg className="w-5 h-5 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h3 className="font-bold text-gray-800">Vastu Score</h3>
            <p className="text-xs text-gray-500">Compliance rating</p>
          </div>
        </div>
      </div>

      <div className="p-5">
        {/* Circular Score */}
        <div className="flex justify-center mb-6">
          <div className={`relative group cursor-default transition-transform duration-300 hover:scale-110 ${score.total >= 90 ? 'animate-pulse-slow' : ''}`}>
            {/* Glow effect for high scores */}
            {score.total >= 80 && (
              <div className="absolute inset-0 rounded-full bg-green-400/20 blur-xl animate-pulse" />
            )}
            <svg className="w-28 h-28 transform -rotate-90 relative z-10">
              {/* Background circle */}
              <circle
                cx="56"
                cy="56"
                r={radius}
                stroke="#E5E7EB"
                strokeWidth="8"
                fill="none"
              />
              {/* Progress circle with animation */}
              <circle
                cx="56"
                cy="56"
                r={radius}
                className={`${getScoreRingColor(score.total)} transition-all duration-1000 ease-out`}
                strokeWidth="8"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
              />
            </svg>
            {/* Score text with counter animation */}
            <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
              <span className={`text-3xl font-bold ${getScoreColor(score.total)} transition-all duration-300 group-hover:scale-110`}>
                {animatedScore}
              </span>
              <span className="text-xs text-gray-400">/ 100</span>
            </div>
          </div>
        </div>

        {/* Status Badge with bounce animation */}
        <div className="flex justify-center mb-6">
          <div className={`px-4 py-1.5 rounded-full bg-gradient-to-r ${getScoreBg(score.total)} text-white text-sm font-semibold shadow-md transform transition-all duration-500 hover:scale-105 hover:shadow-lg animate-bounce-in ${score.total >= 80 ? 'animate-glow' : ''}`}>
            {score.total >= 80 ? '✨ Excellent' : score.total >= 60 ? '👍 Good' : '⚠️ Needs Work'}
          </div>
        </div>

        {/* Score Breakdown with staggered animations */}
        <div className={`space-y-3 transition-all duration-500 ${showBreakdown ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          {scoreItems.map(({ label, value, icon }, index) => (
            <div
              key={label}
              className="group p-2 rounded-lg hover:bg-gray-50 transition-all duration-300 hover:scale-[1.02] cursor-default"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-sm text-gray-600 flex items-center gap-2">
                  <span className="transition-transform duration-300 group-hover:scale-125">{icon}</span>
                  {label}
                </span>
                <span className={`text-sm font-bold ${getScoreColor(value)} transition-all duration-300 group-hover:scale-110`}>
                  {value}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full bg-gradient-to-r ${getScoreBg(value)} rounded-full transition-all duration-1000 ease-out`}
                  style={{
                    width: showBreakdown ? `${value}%` : '0%',
                    transitionDelay: `${index * 150}ms`
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>

        {/* Suggestions with slide-in animation */}
        {score.suggestions.length > 0 && (
          <div className={`mt-6 pt-5 border-t border-gray-100 transition-all duration-700 ${showSuggestions ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
            <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-amber-500 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              Suggestions
              <span className="ml-auto text-xs text-amber-500 bg-amber-100 px-2 py-0.5 rounded-full">
                {score.suggestions.length}
              </span>
            </h4>
            <ul className="space-y-2">
              {score.suggestions.map((suggestion, index) => (
                <li
                  key={index}
                  className={`flex items-start gap-2.5 text-sm text-gray-600 p-2.5 bg-amber-50 rounded-lg border border-amber-100
                    transition-all duration-300 hover:bg-amber-100 hover:border-amber-200 hover:scale-[1.02] cursor-default
                    animate-fade-in-up`}
                  style={{ animationDelay: `${index * 100 + 1000}ms` }}
                >
                  <span className="text-amber-500 mt-0.5 flex-shrink-0 transition-transform duration-300 hover:scale-110">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </span>
                  <span>{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
