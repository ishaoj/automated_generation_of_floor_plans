import { useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { getPreview, PreviewResponse, PreviewSample } from '@/utils/api';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Room colour legend (matches SemanticRenderer palette) ────────────────────
const ROOM_COLORS: Record<string, string> = {
  bedroom:        '#FFC864',
  master_bedroom: '#FF8C32',
  bathroom:       '#64B4FF',
  kitchen:        '#FFFF50',
  living_room:    '#50DC78',
  dining:         '#A0DCA0',
  puja_room:      '#DC64DC',
  parking:        '#B4B4B4',
  balcony:        '#A0DCF0',
  staircase:      '#FF8C8C',
  store:          '#B4966E',
  entrance:       '#FFF0C8',
  ots:            '#C8FFC8',
  corridor:       '#F0DCC0',
  utility:        '#C8BEA0',
};

// ── Types ─────────────────────────────────────────────────────────────────────
interface GeneratePreviewResponse {
  epoch:         number;
  best_val_loss: number;
  semantic_map:  string;
  generated:     string;
  rooms:         Array<{ room_type: string; x: number; y: number; width: number; height: number; area: number }>;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function DatasetSampleCard({ sample, index }: { sample: PreviewSample; index: number }) {
  const cols = [
    { label: 'Semantic Input', src: sample.semantic_map, badge: 'bg-blue-100 text-blue-700' },
    { label: 'Pix2Pix Output', src: sample.generated,    badge: 'bg-orange-100 text-orange-700' },
    { label: 'Ground Truth',   src: sample.target!,      badge: 'bg-green-100 text-green-700' },
  ];
  return (
    <div className="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-100 bg-gray-50 flex items-center gap-2">
        <span className="w-6 h-6 rounded-full bg-vastu-secondary/10 text-vastu-secondary text-xs font-bold flex items-center justify-center">{index + 1}</span>
        <span className="text-sm font-medium text-gray-700">Validation sample {index + 1}</span>
      </div>
      <div className="grid grid-cols-3 gap-4 p-4">
        {cols.map(({ label, src, badge }) => (
          <div key={label} className="flex flex-col items-center gap-2">
            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${badge}`}>{label}</span>
            <img src={`data:image/png;base64,${src}`} alt={label}
              className="w-full aspect-square object-contain rounded-xl border border-gray-100 shadow-sm" />
          </div>
        ))}
      </div>
    </div>
  );
}

function RoomLegend({ rooms }: { rooms: GeneratePreviewResponse['rooms'] }) {
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {rooms.map((r, i) => (
        <div key={i} className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-gray-50 border border-gray-100 text-xs">
          <span className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: ROOM_COLORS[r.room_type] || '#ccc' }} />
          <span className="text-gray-600 capitalize">{r.room_type.replace(/_/g, ' ')}</span>
          <span className="text-gray-400">{r.area.toFixed(0)} ft²</span>
        </div>
      ))}
    </div>
  );
}

// ── Default form state ────────────────────────────────────────────────────────
const DEFAULT_FORM = {
  plot_length: 40,
  plot_width: 30,
  facing_direction: 'north' as const,
  num_bedrooms: 2,
  num_bathrooms: 1,
  include_kitchen: true,
  include_living_room: true,
  include_dining: false,
  include_puja_room: false,
  include_parking: false,
  include_balcony: false,
  include_staircase: false,
  include_ots: false,
  num_variations: 1,
  plot_type: 'open_all' as const,
};

// ── Main page ─────────────────────────────────────────────────────────────────
export default function PreviewPage() {
  const [tab, setTab]   = useState<'request' | 'dataset'>('request');

  // "From request" state
  const [form, setForm]           = useState(DEFAULT_FORM);
  const [reqResult, setReqResult] = useState<GeneratePreviewResponse | null>(null);
  const [reqLoading, setReqLoading] = useState(false);
  const [reqError, setReqError]   = useState<string | null>(null);

  // "Dataset samples" state
  const [dsData, setDsData]       = useState<PreviewResponse | null>(null);
  const [dsLoading, setDsLoading] = useState(false);
  const [dsError, setDsError]     = useState<string | null>(null);
  const [numSamples, setNumSamples] = useState(4);

  // ── Handlers ────────────────────────────────────────────────────────────────
  const handleGenerateFromRequest = async () => {
    setReqLoading(true); setReqError(null);
    try {
      const { data } = await axios.post<GeneratePreviewResponse>(
        `${API_BASE}/api/preview/from-request`, form, { timeout: 60000 }
      );
      setReqResult(data);
    } catch (e: any) {
      setReqError(e.response?.data?.detail || e.message || 'Failed');
    } finally {
      setReqLoading(false);
    }
  };

  const handleLoadDataset = async () => {
    setDsLoading(true); setDsError(null);
    try {
      const result = await getPreview(numSamples);
      setDsData(result);
    } catch (e: any) {
      setDsError(e.message);
    } finally {
      setDsLoading(false);
    }
  };

  const setField = (key: string, val: any) => setForm(f => ({ ...f, [key]: val }));

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>Pix2Pix Preview | Vastu-AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-orange-50/30">
        {/* Header */}
        <header className="bg-gradient-to-r from-vastu-primary to-slate-800 text-white shadow-xl">
          <div className="container mx-auto px-4 py-5 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gradient-to-br from-vastu-secondary to-orange-600 rounded-xl flex items-center justify-center shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold">Pix2Pix Training Preview</h1>
                <p className="text-gray-300 text-sm">Inspect generated floor plan quality</p>
              </div>
            </div>
            <Link href="/" className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm transition-all duration-200">
              ← Back to Generator
            </Link>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8 max-w-6xl">
          {/* Tabs */}
          <div className="flex gap-2 mb-6">
            {[
              { key: 'request', label: 'Your Room Requirements' },
              { key: 'dataset', label: 'Validation Samples' },
            ].map(({ key, label }) => (
              <button key={key}
                onClick={() => setTab(key as any)}
                className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                  tab === key
                    ? 'bg-vastu-secondary text-white shadow-sm'
                    : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* ── TAB: from-request ── */}
          {tab === 'request' && (
            <div className="grid lg:grid-cols-12 gap-6">
              {/* Form */}
              <div className="lg:col-span-4">
                <div className="bg-white rounded-2xl shadow border border-gray-100 p-5 sticky top-6">
                  <h2 className="text-sm font-semibold text-gray-700 mb-4">Room Requirements</h2>

                  <div className="space-y-3">
                    {/* Plot */}
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Length (ft)</label>
                        <input type="number" min={16} max={200} value={form.plot_length}
                          onChange={e => setField('plot_length', Number(e.target.value))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40" />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Width (ft)</label>
                        <input type="number" min={16} max={200} value={form.plot_width}
                          onChange={e => setField('plot_width', Number(e.target.value))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40" />
                      </div>
                    </div>

                    {/* Facing */}
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Facing Direction</label>
                      <select value={form.facing_direction} onChange={e => setField('facing_direction', e.target.value)}
                        className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40">
                        {['north','south','east','west'].map(d => <option key={d} value={d}>{d.charAt(0).toUpperCase()+d.slice(1)}</option>)}
                      </select>
                    </div>

                    {/* Bedrooms / Bathrooms */}
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Bedrooms</label>
                        <select value={form.num_bedrooms} onChange={e => setField('num_bedrooms', Number(e.target.value))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40">
                          {[1,2,3,4].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Bathrooms</label>
                        <select value={form.num_bathrooms} onChange={e => setField('num_bathrooms', Number(e.target.value))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40">
                          {[1,2,3].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                    </div>

                    {/* Plot type */}
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Plot Type</label>
                      <select value={form.plot_type} onChange={e => setField('plot_type', e.target.value)}
                        className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40">
                        <option value="open_all">Open All Sides</option>
                        <option value="open_one">Open One Side</option>
                        <option value="open_two_corner">Open Corner</option>
                      </select>
                    </div>

                    {/* Toggles */}
                    <div>
                      <label className="block text-xs text-gray-500 mb-2">Include Rooms</label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {([
                          ['include_kitchen',     'Kitchen'],
                          ['include_living_room', 'Living Room'],
                          ['include_dining',      'Dining'],
                          ['include_puja_room',   'Puja Room'],
                          ['include_parking',     'Parking'],
                          ['include_balcony',     'Balcony'],
                          ['include_staircase',   'Staircase'],
                          ['include_ots',         'Open to Sky'],
                        ] as [string, string][]).map(([key, label]) => (
                          <label key={key} className="flex items-center gap-2 cursor-pointer group">
                            <div className={`w-8 h-4 rounded-full transition-colors duration-200 relative flex-shrink-0 ${
                              (form as any)[key] ? 'bg-vastu-secondary' : 'bg-gray-200'
                            }`} onClick={() => setField(key, !(form as any)[key])}>
                              <div className={`w-3 h-3 bg-white rounded-full absolute top-0.5 transition-transform duration-200 ${
                                (form as any)[key] ? 'translate-x-4' : 'translate-x-0.5'
                              }`} />
                            </div>
                            <span className="text-xs text-gray-600 group-hover:text-gray-800">{label}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <button onClick={handleGenerateFromRequest} disabled={reqLoading}
                      className="w-full py-2.5 bg-vastu-secondary text-white rounded-xl font-medium text-sm hover:bg-orange-600 disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-200 shadow-sm mt-2">
                      {reqLoading ? 'Generating…' : 'Generate Preview'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Result */}
              <div className="lg:col-span-8">
                {reqError && (
                  <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl mb-5 flex gap-3 items-start">
                    <svg className="w-5 h-5 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div><p className="font-medium">Error</p><p className="text-sm mt-0.5 text-red-600">{reqError}</p></div>
                  </div>
                )}

                {reqLoading && (
                  <div className="bg-white rounded-2xl shadow border border-gray-100 flex flex-col items-center justify-center min-h-[400px] animate-pulse">
                    <div className="w-16 h-16 border-4 border-vastu-secondary/20 border-t-vastu-secondary rounded-full animate-spin mb-4" />
                    <p className="text-gray-500 text-sm">Running constraint solver + Pix2Pix…</p>
                  </div>
                )}

                {!reqLoading && reqResult && (
                  <div className="space-y-4">
                    {/* Epoch badge */}
                    <div className="flex flex-wrap gap-3 items-center">
                      <span className="px-3 py-1 bg-vastu-secondary/10 text-vastu-secondary font-bold rounded-lg text-sm">
                        Epoch {reqResult.epoch}
                      </span>
                      <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-lg text-sm font-mono">
                        best loss {reqResult.best_val_loss}
                      </span>
                    </div>

                    {/* Images side by side */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white rounded-2xl shadow border border-gray-100 p-4">
                        <p className="text-xs font-semibold text-blue-700 bg-blue-50 rounded-lg px-3 py-1 text-center mb-3">Semantic Input (Room Layout)</p>
                        <img src={`data:image/png;base64,${reqResult.semantic_map}`}
                          alt="Semantic map" className="w-full aspect-square object-contain rounded-xl" />
                      </div>
                      <div className="bg-white rounded-2xl shadow border border-gray-100 p-4">
                        <p className="text-xs font-semibold text-orange-700 bg-orange-50 rounded-lg px-3 py-1 text-center mb-3">Pix2Pix Generated Output</p>
                        <img src={`data:image/png;base64,${reqResult.generated}`}
                          alt="Generated" className="w-full aspect-square object-contain rounded-xl" />
                      </div>
                    </div>

                    {/* Room legend */}
                    <div className="bg-white rounded-xl border border-gray-100 p-4">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Room Legend</p>
                      <RoomLegend rooms={reqResult.rooms} />
                    </div>
                  </div>
                )}

                {!reqLoading && !reqResult && !reqError && (
                  <div className="bg-white rounded-2xl shadow border border-gray-100 flex flex-col items-center justify-center min-h-[400px]">
                    <div className="w-20 h-20 bg-orange-50 rounded-full flex items-center justify-center mb-4">
                      <svg className="w-10 h-10 text-vastu-secondary/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                          d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-700 mb-1">Ready to Preview</h3>
                    <p className="text-gray-400 text-sm text-center max-w-xs">
                      Fill in room requirements on the left and click <strong>Generate Preview</strong> to see what Pix2Pix produces for your layout.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── TAB: dataset samples ── */}
          {tab === 'dataset' && (
            <div>
              <div className="bg-white rounded-2xl shadow border border-gray-100 p-5 mb-6">
                <div className="flex flex-wrap items-end gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Samples</label>
                    <select value={numSamples} onChange={e => setNumSamples(Number(e.target.value))}
                      className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-vastu-secondary/40">
                      {[2,4,6,8].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </div>
                  <button onClick={handleLoadDataset} disabled={dsLoading}
                    className="px-6 py-2 bg-vastu-secondary text-white rounded-xl font-medium text-sm hover:bg-orange-600 disabled:opacity-60 transition-all">
                    {dsLoading ? 'Loading…' : 'Load Samples'}
                  </button>
                  {dsData && (
                    <div className="flex flex-wrap gap-3 items-center">
                      <span className="px-3 py-1 bg-vastu-secondary/10 text-vastu-secondary font-bold rounded-lg text-sm">Epoch {dsData.epoch}</span>
                      <span className="px-3 py-1 bg-gray-100 text-gray-600 font-mono rounded-lg text-sm">loss {dsData.best_val_loss}</span>
                    </div>
                  )}
                </div>
              </div>

              {dsError && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl mb-5">
                  {dsError}
                </div>
              )}

              {dsLoading && (
                <div className="space-y-6">
                  {Array.from({ length: numSamples }).map((_, i) => (
                    <div key={i} className="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden animate-pulse">
                      <div className="h-12 bg-gray-100" />
                      <div className="grid grid-cols-3 gap-4 p-4">
                        {[0,1,2].map(j => <div key={j} className="aspect-square bg-gray-100 rounded-xl" />)}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {!dsLoading && dsData && (
                <div className="space-y-6">
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-blue-200" />Semantic input (dataset SVG)</span>
                    <span>→</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-orange-200" />Pix2Pix generated</span>
                    <span>→</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-green-200" />Ground truth F1_scaled.png</span>
                  </div>
                  {dsData.samples.map((s, i) => <DatasetSampleCard key={i} sample={s} index={i} />)}
                </div>
              )}

              {!dsLoading && !dsData && !dsError && (
                <div className="bg-white rounded-2xl shadow border border-gray-100 flex flex-col items-center justify-center min-h-[300px]">
                  <p className="text-gray-400 text-sm">Click <strong>Load Samples</strong> to compare Pix2Pix output against ground-truth floor plans.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </>
  );
}
