import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import FloorPlanForm from '@/components/FloorPlanForm';
import PlanGallery from '@/components/PlanGallery';
import { GeneratedPlan, FloorPlanRequest } from '@/utils/types';
import { generateFloorPlans } from '@/utils/api';

export default function Home() {
  const [plans, setPlans] = useState<GeneratedPlan[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestSummary, setRequestSummary] = useState<{plot_size: string; facing: string; bedrooms: number; bathrooms: number} | null>(null);
  const [loadingStep, setLoadingStep] = useState(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Animated loading steps
  useEffect(() => {
    if (loading) {
      const steps = ['Analyzing requirements...', 'Applying Vastu rules...', 'Optimizing layout...', 'Rendering floor plans...'];
      let step = 0;
      const interval = setInterval(() => {
        step = (step + 1) % steps.length;
        setLoadingStep(step);
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [loading]);

  const loadingSteps = [
    'Analyzing requirements...',
    'Applying Vastu rules...',
    'Optimizing layout...',
    'Rendering floor plans...'
  ];

  const handleGenerate = async (request: FloorPlanRequest) => {
    setLoading(true);
    setError(null);
    setLoadingStep(0);

    try {
      const result = await generateFloorPlans(request);
      setPlans(result.plans);
      setRequestSummary(result.request_summary);
    } catch (err: any) {
      setError(err.message || 'Failed to generate floor plans');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Vastu-AI | AI-Powered Floor Plan Generator</title>
        <meta name="description" content="Generate Vastu-compliant floor plans using AI" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-orange-50/30">
        {/* Animated Header */}
        <header className={`bg-gradient-to-r from-vastu-primary to-slate-800 text-white shadow-xl transition-all duration-500 ${mounted ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'}`}>
          <div className="container mx-auto px-4 py-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                {/* Animated Logo */}
                <div className="w-12 h-12 bg-gradient-to-br from-vastu-secondary to-orange-600 rounded-xl flex items-center justify-center shadow-lg hover:scale-110 transition-transform duration-300 cursor-pointer group">
                  <svg className="w-7 h-7 text-white group-hover:animate-wiggle" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-2xl font-bold tracking-tight gradient-text">Vastu-AI</h1>
                  <p className="text-gray-300 text-sm">AI-Powered Floor Plan Generator</p>
                </div>
              </div>

              {/* Nav links */}
              <div className="hidden md:flex items-center gap-3 text-sm text-gray-300">
                <Link
                  href="/preview"
                  className="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg text-xs font-medium transition-all duration-200 flex items-center gap-1.5"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" />
                  </svg>
                  Pix2Pix Preview
                </Link>

              {/* Animated compass indicator */}
                <div className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-xl backdrop-blur hover:bg-white/20 transition-all duration-300 cursor-default group">
                  <svg className="w-5 h-5 text-vastu-secondary group-hover:animate-spin-slow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  <span>Vastu Shastra Compliant</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8">
          <div className="grid lg:grid-cols-12 gap-8">
            {/* Left Panel - Form with slide-in animation */}
            <div className={`lg:col-span-4 xl:col-span-3 transition-all duration-700 delay-100 ${mounted ? 'translate-x-0 opacity-100' : '-translate-x-8 opacity-0'}`}>
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden sticky top-6 card-hover">
                {/* Form Header */}
                <div className="bg-gradient-to-r from-gray-50 to-orange-50 px-6 py-4 border-b border-gray-100">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-vastu-secondary/10 rounded-xl flex items-center justify-center float-icon">
                      <svg className="w-5 h-5 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-gray-800">Floor Plan Details</h2>
                      <p className="text-xs text-gray-500">Enter your requirements below</p>
                    </div>
                  </div>
                </div>

                <div className="p-6">
                  <FloorPlanForm onSubmit={handleGenerate} isLoading={loading} />
                </div>
              </div>
            </div>

            {/* Right Panel - Results with fade-in animation */}
            <div className={`lg:col-span-8 xl:col-span-9 transition-all duration-700 delay-200 ${mounted ? 'translate-x-0 opacity-100' : 'translate-x-8 opacity-0'}`}>
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl mb-6 flex items-start gap-3 animate-fade-in-down">
                  <svg className="w-5 h-5 mt-0.5 flex-shrink-0 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <p className="font-medium">Generation Failed</p>
                    <p className="text-sm mt-1 text-red-600">{error}</p>
                  </div>
                </div>
              )}

              {loading ? (
                <div className="flex flex-col items-center justify-center min-h-[500px] bg-white rounded-2xl shadow-lg border border-gray-100 animate-fade-in">
                  {/* Enhanced animated loading */}
                  <div className="relative">
                    {/* Outer ring */}
                    <div className="w-24 h-24 border-4 border-vastu-secondary/10 rounded-full"></div>
                    {/* Spinning ring */}
                    <div className="w-24 h-24 border-4 border-transparent border-t-vastu-secondary border-r-vastu-secondary rounded-full animate-spin absolute top-0 left-0"></div>
                    {/* Pulsing center */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-16 h-16 bg-gradient-to-br from-orange-100 to-orange-50 rounded-full flex items-center justify-center animate-pulse">
                        <svg className="w-8 h-8 text-vastu-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                        </svg>
                      </div>
                    </div>
                    {/* Orbiting dots */}
                    <div className="absolute inset-0 animate-spin-slow">
                      <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1 w-2 h-2 bg-vastu-secondary rounded-full"></div>
                    </div>
                  </div>

                  <p className="mt-8 text-gray-700 font-semibold text-lg">Generating floor plans...</p>

                  {/* Animated step indicator */}
                  <div className="mt-4 flex items-center gap-2">
                    <div className="loading-dots flex gap-1">
                      <span className="w-2 h-2 bg-vastu-secondary rounded-full"></span>
                      <span className="w-2 h-2 bg-vastu-secondary rounded-full"></span>
                      <span className="w-2 h-2 bg-vastu-secondary rounded-full"></span>
                    </div>
                    <span className="text-sm text-gray-500 min-w-[180px] transition-all duration-300">
                      {loadingSteps[loadingStep]}
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div className="mt-6 w-64 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-vastu-secondary to-orange-400 rounded-full animate-pulse" style={{width: '70%'}}></div>
                  </div>

                  {/* Loading tips with rotation */}
                  <div className="mt-8 px-6 py-4 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl max-w-md text-center border border-orange-100">
                    <p className="text-sm text-gray-600">
                      <span className="font-semibold text-vastu-secondary">Did you know?</span> Vastu Shastra is an ancient Indian science that harmonizes architecture with natural and cosmic energies.
                    </p>
                  </div>
                </div>
              ) : plans.length > 0 ? (
                <div className="animate-fade-in-up">
                  <PlanGallery plans={plans} requestSummary={requestSummary} />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center min-h-[500px] bg-white rounded-2xl shadow-lg border border-gray-100 animate-fade-in">
                  {/* Animated empty state illustration */}
                  <div className="w-32 h-32 bg-gradient-to-br from-orange-100 to-orange-50 rounded-full flex items-center justify-center mb-6 animate-float">
                    <svg className="w-16 h-16 text-vastu-secondary/60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">Ready to Design</h3>
                  <p className="text-gray-500 text-center max-w-md mb-8">
                    Enter your plot dimensions and room requirements, then click Generate to create Vastu-compliant floor plans.
                  </p>

                  {/* Animated quick tips */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl">
                    {[
                      { num: '1', color: 'blue', title: 'Set Plot Size', desc: 'Enter length & width' },
                      { num: '2', color: 'green', title: 'Choose Rooms', desc: 'Select what you need' },
                      { num: '3', color: 'orange', title: 'Generate', desc: 'Get AI-powered plans' }
                    ].map((step, i) => (
                      <div
                        key={step.num}
                        className={`flex items-start gap-3 p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-all duration-300 hover:scale-105 hover:shadow-md cursor-default animate-fade-in-up`}
                        style={{ animationDelay: `${i * 150}ms` }}
                      >
                        <div className={`w-8 h-8 bg-${step.color}-100 rounded-lg flex items-center justify-center flex-shrink-0`}>
                          <span className={`text-${step.color}-600 font-bold text-sm`}>{step.num}</span>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-700">{step.title}</p>
                          <p className="text-xs text-gray-500">{step.desc}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Animated Footer */}
        <footer className={`bg-white border-t mt-12 transition-all duration-700 delay-300 ${mounted ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'}`}>
          <div className="container mx-auto px-4 py-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3 group cursor-default">
                <div className="w-8 h-8 bg-gradient-to-br from-vastu-secondary to-orange-600 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700">Vastu-AI</p>
                  <p className="text-xs text-gray-500">Final Year Project</p>
                </div>
              </div>
              <p className="text-sm text-gray-500">
                Generating Vastu-compliant floor plans using AI & constraint optimization
              </p>
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
