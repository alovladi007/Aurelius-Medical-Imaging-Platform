'use client'

import { useState, useEffect } from 'react'
import { Search, Filter, Save, Download, X, Calendar, Building, Activity } from 'lucide-react'

interface SearchFilters {
  query: string
  modalities: string[]
  bodyParts: string[]
  dateFrom: string
  dateTo: string
  institutions: string[]
  hasAnnotations: boolean | null
  annotationLabels: string[]
  predictionModels: string[]
  semanticSearch: boolean
}

interface SearchResult {
  study_id: string
  study_instance_uid: string
  study_date: string
  study_description: string
  modality: string
  body_part: string
  number_of_series: number
  number_of_instances: number
  score: number
  highlights: Record<string, string[]>
}

interface Facets {
  modalities: string[]
  body_parts: string[]
  institutions: string[]
  annotation_labels: string[]
  prediction_models: string[]
}

export default function SearchPage() {
  const [filters, setFilters] = useState<SearchFilters>({
    query: '',
    modalities: [],
    bodyParts: [],
    dateFrom: '',
    dateTo: '',
    institutions: [],
    hasAnnotations: null,
    annotationLabels: [],
    predictionModels: [],
    semanticSearch: false
  })

  const [results, setResults] = useState<SearchResult[]>([])
  const [facets, setFacets] = useState<Facets | null>(null)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [showFilters, setShowFilters] = useState(true)
  const [aggregations, setAggregations] = useState<any>({})

  // Load available facets on mount
  useEffect(() => {
    loadFacets()
  }, [])

  const loadFacets = async () => {
    try {
      const response = await fetch('http://localhost:8004/facets')
      const data = await response.json()
      setFacets(data)
    } catch (error) {
      console.error('Failed to load facets:', error)
    }
  }

  const handleSearch = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8004/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...filters,
          date_from: filters.dateFrom || null,
          date_to: filters.dateTo || null,
          has_annotations: filters.hasAnnotations,
          annotation_labels: filters.annotationLabels.length > 0 ? filters.annotationLabels : null,
          prediction_models: filters.predictionModels.length > 0 ? filters.predictionModels : null,
          semantic_search: filters.semanticSearch,
          page,
          page_size: 20
        })
      })

      const data = await response.json()
      setResults(data.results)
      setTotal(data.total)
      setAggregations(data.aggregations)
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleExportCSV = async () => {
    try {
      const response = await fetch('http://localhost:8004/export/csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filters)
      })
      // Download CSV
      alert('CSV export initiated')
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const toggleFilter = (filterType: keyof SearchFilters, value: string) => {
    const currentValues = filters[filterType] as string[]
    setFilters({
      ...filters,
      [filterType]: currentValues.includes(value)
        ? currentValues.filter(v => v !== value)
        : [...currentValues, value]
    })
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Study Search</h1>
          <p className="text-gray-600">Search across all medical imaging studies with advanced filtering</p>
        </div>

        {/* Search Bar */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
              <input
                type="text"
                value={filters.query}
                onChange={(e) => setFilters({ ...filters, query: e.target.value })}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search studies, descriptions, physicians..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
            >
              <Search className="h-5 w-5" />
              {loading ? 'Searching...' : 'Search'}
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
            >
              <Filter className="h-5 w-5" />
              Filters
            </button>
          </div>

          {/* Semantic Search Toggle */}
          <div className="mt-4 flex items-center gap-2">
            <input
              type="checkbox"
              id="semantic"
              checked={filters.semanticSearch}
              onChange={(e) => setFilters({ ...filters, semanticSearch: e.target.checked })}
              className="rounded border-gray-300"
            />
            <label htmlFor="semantic" className="text-sm text-gray-700">
              Enable semantic search (AI-powered similarity)
            </label>
          </div>
        </div>

        <div className="flex gap-6">
          {/* Filters Sidebar */}
          {showFilters && facets && (
            <div className="w-80 bg-white rounded-lg shadow-sm p-6 h-fit">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Filters</h2>
                <button
                  onClick={() => setFilters({
                    query: '',
                    modalities: [],
                    bodyParts: [],
                    dateFrom: '',
                    dateTo: '',
                    institutions: [],
                    hasAnnotations: null,
                    annotationLabels: [],
                    predictionModels: [],
                    semanticSearch: false
                  })}
                  className="text-sm text-blue-600 hover:text-blue-700"
                >
                  Clear all
                </button>
              </div>

              {/* Date Range */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-900 mb-2 flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Date Range
                </h3>
                <div className="space-y-2">
                  <input
                    type="date"
                    value={filters.dateFrom}
                    onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    placeholder="From"
                  />
                  <input
                    type="date"
                    value={filters.dateTo}
                    onChange={(e) => setFilters({ ...filters, dateTo: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    placeholder="To"
                  />
                </div>
              </div>

              {/* Modality */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-900 mb-2 flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Modality
                </h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {facets.modalities.map((modality) => (
                    <label key={modality} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={filters.modalities.includes(modality)}
                        onChange={() => toggleFilter('modalities', modality)}
                        className="rounded border-gray-300"
                      />
                      <span className="text-gray-700">{modality}</span>
                      {aggregations.modalities && (
                        <span className="ml-auto text-xs text-gray-500">
                          ({aggregations.modalities.find((a: any) => a.key === modality)?.count || 0})
                        </span>
                      )}
                    </label>
                  ))}
                </div>
              </div>

              {/* Body Part */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-900 mb-2">Body Part</h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {facets.body_parts.slice(0, 10).map((bodyPart) => (
                    <label key={bodyPart} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={filters.bodyParts.includes(bodyPart)}
                        onChange={() => toggleFilter('bodyParts', bodyPart)}
                        className="rounded border-gray-300"
                      />
                      <span className="text-gray-700">{bodyPart}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Institution */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-900 mb-2 flex items-center gap-2">
                  <Building className="h-4 w-4" />
                  Institution
                </h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {facets.institutions.map((institution) => (
                    <label key={institution} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={filters.institutions.includes(institution)}
                        onChange={() => toggleFilter('institutions', institution)}
                        className="rounded border-gray-300"
                      />
                      <span className="text-gray-700">{institution}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Annotations */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-900 mb-2">Annotations</h3>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={filters.hasAnnotations === true}
                      onChange={() => setFilters({ ...filters, hasAnnotations: filters.hasAnnotations ? null : true })}
                      className="rounded border-gray-300"
                    />
                    <span className="text-gray-700">Has annotations</span>
                  </label>
                  {facets.annotation_labels.length > 0 && (
                    <div className="ml-6 space-y-2">
                      {facets.annotation_labels.map((label) => (
                        <label key={label} className="flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={filters.annotationLabels.includes(label)}
                            onChange={() => toggleFilter('annotationLabels', label)}
                            className="rounded border-gray-300"
                          />
                          <span className="text-gray-600 text-xs">{label}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* AI Predictions */}
              {facets.prediction_models.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-gray-900 mb-2">AI Models</h3>
                  <div className="space-y-2">
                    {facets.prediction_models.map((model) => (
                      <label key={model} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={filters.predictionModels.includes(model)}
                          onChange={() => toggleFilter('predictionModels', model)}
                          className="rounded border-gray-300"
                        />
                        <span className="text-gray-700 text-xs">{model}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-2">
                <button
                  onClick={handleExportCSV}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center justify-center gap-2 text-sm"
                >
                  <Download className="h-4 w-4" />
                  Export to CSV
                </button>
                <button
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center justify-center gap-2 text-sm"
                >
                  <Save className="h-4 w-4" />
                  Save Query
                </button>
              </div>
            </div>
          )}

          {/* Results */}
          <div className="flex-1">
            {/* Results Header */}
            <div className="bg-white rounded-lg shadow-sm p-4 mb-4 flex items-center justify-between">
              <div className="text-sm text-gray-600">
                {total > 0 ? (
                  <span>Found <strong>{total.toLocaleString()}</strong> studies</span>
                ) : (
                  <span>No results found</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page === 1}
                  className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
                >
                  Previous
                </button>
                <span className="text-sm text-gray-600">Page {page}</span>
                <button
                  onClick={() => setPage(page + 1)}
                  disabled={results.length < 20}
                  className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>

            {/* Results List */}
            <div className="space-y-4">
              {results.map((result) => (
                <div key={result.study_id} className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-900 mb-1">
                        {result.study_description || 'Untitled Study'}
                      </h3>
                      <div className="flex items-center gap-4 text-sm text-gray-600">
                        <span className="font-medium">{result.modality}</span>
                        <span>•</span>
                        <span>{result.body_part || 'Unknown'}</span>
                        <span>•</span>
                        <span>{new Date(result.study_date).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <div className="text-sm text-gray-500">
                      Score: {result.score.toFixed(2)}
                    </div>
                  </div>

                  <div className="mt-3 text-sm text-gray-600">
                    <div className="flex items-center gap-4">
                      <span>{result.number_of_series} series</span>
                      <span>•</span>
                      <span>{result.number_of_instances} instances</span>
                    </div>
                  </div>

                  {/* Highlights */}
                  {Object.keys(result.highlights).length > 0 && (
                    <div className="mt-3 p-3 bg-yellow-50 rounded text-sm">
                      {Object.entries(result.highlights).map(([field, highlights]) => (
                        <div key={field}>
                          <strong className="text-gray-700">{field}:</strong>{' '}
                          <span dangerouslySetInnerHTML={{ __html: highlights[0] }} />
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="mt-4 flex items-center gap-2">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">
                      View Study
                    </button>
                    <button className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-50 text-sm">
                      Add to Worklist
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {loading && (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-gray-600">Searching...</p>
              </div>
            )}

            {!loading && results.length === 0 && filters.query && (
              <div className="text-center py-12 bg-white rounded-lg shadow-sm">
                <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No studies found</h3>
                <p className="text-gray-600">Try adjusting your search query or filters</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
