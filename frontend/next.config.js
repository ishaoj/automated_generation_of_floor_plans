/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Rewrites only apply in local development.
  // In production (Netlify), the frontend calls the backend directly
  // via the NEXT_PUBLIC_API_URL environment variable set in Netlify.
  ...(process.env.NODE_ENV === 'development' && {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:8000/api/:path*',
        },
      ];
    },
  }),
};

module.exports = nextConfig;
