import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useStore = create(
  persist(
    (set, get) => ({
      // Predictions history
      predictions: [],
      addPrediction: (prediction) =>
        set((state) => ({
          predictions: [
            {
              ...prediction,
              id: Date.now(),
              timestamp: new Date().toISOString(),
            },
            ...state.predictions,
          ].slice(0, 100), // Keep last 100 predictions
        })),
      clearPredictions: () => set({ predictions: [] }),
      deletePrediction: (id) =>
        set((state) => ({
          predictions: state.predictions.filter((p) => p.id !== id),
        })),

      // Settings
      settings: {
        theme: 'light',
        notifications: true,
        autoSave: true,
        confidenceThreshold: 0.7,
      },
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),

      // UI state
      sidebarOpen: true,
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      // Current prediction state
      currentPrediction: null,
      setCurrentPrediction: (prediction) => set({ currentPrediction: prediction }),
      clearCurrentPrediction: () => set({ currentPrediction: null }),

      // Statistics
      getStatistics: () => {
        const { predictions } = get();
        const total = predictions.length;
        const byType = predictions.reduce((acc, pred) => {
          const type = pred.cancer_type || 'Unknown';
          acc[type] = (acc[type] || 0) + 1;
          return acc;
        }, {});
        const avgConfidence =
          predictions.reduce((sum, pred) => sum + (pred.confidence || 0), 0) /
          (total || 1);

        return {
          total,
          byType,
          avgConfidence,
          highRiskCount: predictions.filter((p) => p.risk_score > 0.7).length,
        };
      },
    }),
    {
      name: 'cancer-ai-storage',
      partialize: (state) => ({
        predictions: state.predictions,
        settings: state.settings,
      }),
    }
  )
);

export default useStore;
