import React, { useState, useMemo } from "react";
import { useAppContext } from "../context/AppContext";
import { StatusBadge } from "../components/ui/StatusBadge";
import { ConfidenceBar } from "../components/ui/ConfidenceBar";
import type { EvidenceItem } from "../types";
import {
  BookOpen,
  Search,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Filter,
  FileText,
  BookMarked,
  Database,
  Globe,
  Bookmark,
} from "lucide-react";

const sourceIcons: Record<string, React.ElementType> = {
  PubMed: Globe,
  Guidelines: BookMarked,
  "ICD-10": Database,
  "Local KB": FileText,
  UpToDate: Bookmark,
};

const sourceColors: Record<string, string> = {
  PubMed: "bg-blue-50 text-blue-700 border-blue-200",
  Guidelines: "bg-emerald-50 text-emerald-700 border-emerald-200",
  "ICD-10": "bg-purple-50 text-purple-700 border-purple-200",
  "Local KB": "bg-slate-50 text-slate-700 border-slate-200",
  UpToDate: "bg-orange-50 text-orange-700 border-orange-200",
};

function EvidenceCard({ item }: { item: EvidenceItem }) {
  const [expanded, setExpanded] = useState(false);
  const Icon = sourceIcons[item.source] || BookOpen;

  return (
    <div className="bg-card rounded-xl border border-border overflow-hidden transition-shadow hover:shadow-md">
      <div className="p-5">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border ${sourceColors[item.source] || ""}`}>
                <Icon className="w-3 h-3" />
                {item.source}
              </span>
              {item.metadata.specialty && (
                <StatusBadge variant="neutral">{item.metadata.specialty}</StatusBadge>
              )}
            </div>
            <h4 className="text-[14px] text-foreground leading-snug">{item.title}</h4>
          </div>
          <div className="text-right shrink-0">
            <div className="text-[20px] text-foreground">{Math.round(item.relevanceScore * 100)}</div>
            <div className="text-[10px] text-muted-foreground">relevance</div>
          </div>
        </div>

        <ConfidenceBar value={item.relevanceScore} size="sm" showLabel={false} className="mb-3" />

        {/* Metadata */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground mb-3">
          {item.metadata.authors && <span>{item.metadata.authors}</span>}
          {item.metadata.year && <span>{item.metadata.year}</span>}
          {item.metadata.journal && <span className="italic">{item.metadata.journal}</span>}
        </div>

        {/* Snippet preview / expanded */}
        <p className={`text-[12px] text-muted-foreground leading-relaxed ${expanded ? "" : "line-clamp-2"}`}>
          {item.snippet}
        </p>

        <div className="flex items-center justify-between mt-3 pt-3 border-t border-border">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-[12px] text-primary hover:underline"
          >
            {expanded ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" /> Collapse
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" /> Read more
              </>
            )}
          </button>
          {item.metadata.doi && (
            <a
              href={`https://doi.org/${item.metadata.doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-[11px] text-primary hover:underline"
            >
              <ExternalLink className="w-3 h-3" />
              DOI
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export function KnowledgeEvidence() {
  const { evidence } = useAppContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");

  const allSources = useMemo(
    () => Array.from(new Set(evidence.map((e) => e.source))),
    [evidence]
  );

  const filtered = useMemo(() => {
    let items = evidence;
    if (sourceFilter !== "all") {
      items = items.filter((e) => e.source === sourceFilter);
    }
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      items = items.filter(
        (e) =>
          e.title.toLowerCase().includes(q) ||
          e.snippet.toLowerCase().includes(q) ||
          e.source.toLowerCase().includes(q)
      );
    }
    return items.sort((a, b) => b.relevanceScore - a.relevanceScore);
  }, [evidence, sourceFilter, searchTerm]);

  if (evidence.length === 0) {
    return (
      <div className="p-6 lg:p-8 max-w-7xl mx-auto">
        <h1 className="text-foreground flex items-center gap-2 mb-2">
          <BookOpen className="w-5 h-5 text-primary" />
          Knowledge Evidence
        </h1>
        <div className="text-center py-20 bg-card rounded-xl border border-dashed border-border mt-6">
          <BookOpen className="w-12 h-12 mx-auto text-muted-foreground/30 mb-4" />
          <p className="text-[14px] text-muted-foreground">No evidence retrieved yet</p>
          <p className="text-[12px] text-muted-foreground/70 mt-1">
            Run the analysis pipeline to retrieve relevant medical literature
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-foreground flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-primary" />
          Knowledge Evidence
        </h1>
        <p className="text-[13px] text-muted-foreground mt-1">
          Retrieved literature, guidelines, and knowledge base entries supporting the analysis.
        </p>
      </div>

      {/* Search and filters */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search evidence, titles, snippets..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-border bg-input-background text-[13px] text-foreground"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-muted-foreground" />
          <div className="flex items-center gap-1 bg-secondary rounded-lg p-0.5">
            <button
              onClick={() => setSourceFilter("all")}
              className={`px-3 py-1 rounded-md text-[11px] transition-colors ${
                sourceFilter === "all" ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"
              }`}
            >
              All ({evidence.length})
            </button>
            {allSources.map((s) => (
              <button
                key={s}
                onClick={() => setSourceFilter(s)}
                className={`px-3 py-1 rounded-md text-[11px] transition-colors ${
                  sourceFilter === s ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results count */}
      <p className="text-[12px] text-muted-foreground">
        Showing {filtered.length} of {evidence.length} evidence items
      </p>

      {/* Evidence cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filtered.map((item) => (
          <EvidenceCard key={item.id} item={item} />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-[13px] text-muted-foreground">
          No evidence matches your search criteria
        </div>
      )}
    </div>
  );
}
