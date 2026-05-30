/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Allow manual class toggle or default dark
  theme: {
    extend: {
      colors: {
        clinical: {
          bg: '#070a13',       // Ultra dark navy/slate
          card: '#0f1424',     // Dark slate card
          border: '#1f2945',   // Rich navy border
          text: '#f1f5f9',     // Slate 100
          textMuted: '#94a3b8',// Slate 400
          accent: '#6366f1',   // Indigo 500
          highlight: '#06b6d4',// Cyan 500
          success: '#10b981',  // Emerald 500
          warning: '#f59e0b',  // Amber 500
          danger: '#ef4444',   // Rose 500
        }
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
