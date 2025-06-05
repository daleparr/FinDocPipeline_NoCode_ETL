# This completes the main function for FinDocPipeline_Enhanced.py

                    st.session_state.processing_results = results
        
        with col2:
            if st.button("📊 Show File Details"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"**{file.name}** - {file.size / 1024:.1f} KB - {file.type}")
        
        with col3:
            if st.button("🗑️ Clear Queue"):
                st.rerun()
        
        # Show processing results with detailed data
        if 'processing_results' in st.session_state:
            results = st.session_state.processing_results
            
            st.header("📈 Detailed Processing Results")
            
            # Processing summary
            summary = results.get('processing_summary', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", summary.get('total_documents', 0))
            with col2:
                st.metric("Successful", summary.get('successful_documents', 0))
            with col3:
                st.metric("Failed", summary.get('failed_documents', 0))
            with col4:
                st.metric("Success Rate", f"{summary.get('success_rate', 0):.1%}")
            
            # Detailed row counts
            st.subheader("📊 Detailed Data Generated")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Detailed Rows", f"{summary.get('total_detailed_rows_generated', 0):,}")
            with col2:
                st.metric("NLP Dataset Rows", f"{summary.get('total_nlp_rows_generated', 0):,}")
            with col3:
                st.metric("Financial Metrics", f"{summary.get('total_metrics_extracted', 0):,}")
            
            # Show consolidated detailed data
            consolidated_data = results.get('consolidated_detailed_data', {})
            
            if consolidated_data:
                st.subheader("🔍 Consolidated Detailed Data")
                
                # Show data type tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "NLP Data", "Metrics", "Visual Data"])
                
                with tab1:
                    raw_data = consolidated_data.get('all_detailed_raw_data', [])
                    if raw_data:
                        st.write(f"**{len(raw_data):,} detailed raw data rows across all documents**")
                        
                        # Show sample of raw data
                        if st.checkbox("Show Raw Data Sample (first 100 rows)"):
                            sample_df = pd.DataFrame(raw_data[:100])
                            st.dataframe(sample_df, use_container_width=True)
                        
                        # Export raw data
                        if st.button("📄 Export All Raw Data as CSV"):
                            raw_df = pd.DataFrame(raw_data)
                            csv_data = raw_df.to_csv(index=False)
                            st.download_button(
                                label="Download Raw Data CSV",
                                data=csv_data,
                                file_name=f"multi_doc_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No raw data available")
                
                with tab2:
                    nlp_data = consolidated_data.get('all_detailed_nlp_data', [])
                    if nlp_data:
                        st.write(f"**{len(nlp_data):,} NLP processed rows across all documents**")
                        
                        # Show sample of NLP data
                        if st.checkbox("Show NLP Data Sample (first 100 rows)"):
                            sample_df = pd.DataFrame(nlp_data[:100])
                            st.dataframe(sample_df, use_container_width=True)
                        
                        # Export NLP data
                        if st.button("📄 Export All NLP Data as CSV"):
                            nlp_df = pd.DataFrame(nlp_data)
                            csv_data = nlp_df.to_csv(index=False)
                            st.download_button(
                                label="Download NLP Data CSV",
                                data=csv_data,
                                file_name=f"multi_doc_nlp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No NLP data available")
                
                with tab3:
                    metrics_data = consolidated_data.get('all_detailed_metrics', [])
                    if metrics_data:
                        st.write(f"**{len(metrics_data):,} financial metrics across all documents**")
                        
                        # Show sample of metrics
                        if st.checkbox("Show Metrics Sample"):
                            sample_df = pd.DataFrame(metrics_data)
                            st.dataframe(sample_df, use_container_width=True)
                        
                        # Export metrics
                        if st.button("📄 Export All Metrics as CSV"):
                            metrics_df = pd.DataFrame(metrics_data)
                            csv_data = metrics_df.to_csv(index=False)
                            st.download_button(
                                label="Download Metrics CSV",
                                data=csv_data,
                                file_name=f"multi_doc_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No metrics data available")
                
                with tab4:
                    visual_data = consolidated_data.get('all_detailed_visual_data', [])
                    if visual_data:
                        st.write(f"**{len(visual_data):,} visual content items across all documents**")
                        
                        # Show sample of visual data
                        if st.checkbox("Show Visual Data Sample"):
                            sample_df = pd.DataFrame(visual_data[:50])  # Limit for display
                            st.dataframe(sample_df, use_container_width=True)
                    else:
                        st.info("No visual data available")
            
            # Individual document results
            st.subheader("📋 Individual Document Results")
            individual_results = results.get('individual_document_results', {})
            
            for filename, result in individual_results.items():
                with st.expander(f"📄 {filename} - {result.get('status', 'unknown').title()}"):
                    if result.get('status') == 'completed':
                        detailed_content = result.get('detailed_content', {})
                        summary_stats = detailed_content.get('summary_stats', {})
                        
                        # Show document-specific stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**Processing Info:**")
                            st.write(f"- File Type: {result.get('file_extension', 'unknown').upper()}")
                            st.write(f"- Processing Time: {result.get('processing_time', 0):.2f}s")
                            st.write(f"- Extraction Method: {detailed_content.get('extraction_method', 'unknown')}")
                        
                        with col2:
                            st.write("**Content Stats:**")
                            for key, value in summary_stats.items():
                                if isinstance(value, (int, float)):
                                    st.write(f"- {key.replace('_', ' ').title()}: {value:,}")
                        
                        with col3:
                            st.write("**Classification:**")
                            classification = result.get('classification', {})
                            st.write(f"- Type: {classification.get('document_type', 'unknown')}")
                            st.write(f"- Confidence: {classification.get('confidence', 0):.2f}")
                            st.write(f"- Method: {classification.get('method', 'unknown')}")
                        
                        # Show detailed data for this document
                        if st.checkbox(f"Show detailed data for {filename}", key=f"details_{filename}"):
                            
                            # Raw data for this document
                            if 'detailed_raw_data' in detailed_content:
                                st.write("**Raw Data Rows:**")
                                raw_data = detailed_content['detailed_raw_data']
                                if raw_data:
                                    raw_df = pd.DataFrame(raw_data)
                                    st.dataframe(raw_df, use_container_width=True)
                            
                            # NLP data for this document
                            if 'detailed_nlp_data' in detailed_content:
                                st.write("**NLP Processed Data:**")
                                nlp_data = detailed_content['detailed_nlp_data']
                                if nlp_data:
                                    nlp_df = pd.DataFrame(nlp_data)
                                    st.dataframe(nlp_df, use_container_width=True)
                            
                            # Metrics for this document
                            if 'detailed_metrics' in detailed_content:
                                st.write("**Financial Metrics:**")
                                metrics_data = detailed_content['detailed_metrics']
                                if metrics_data:
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True)
                    
                    elif result.get('status') == 'failed':
                        st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            
            # Export all results
            st.subheader("💾 Export All Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Complete Results as JSON"):
                    json_data = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="Download Complete Results JSON",
                        data=json_data,
                        file_name=f"multi_doc_complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📊 Export Processing Summary as CSV"):
                    # Create summary CSV
                    summary_data = []
                    for filename, result in individual_results.items():
                        if result.get('status') == 'completed':
                            detailed_content = result.get('detailed_content', {})
                            summary_stats = detailed_content.get('summary_stats', {})
                            
                            summary_row = {
                                'filename': filename,
                                'status': result.get('status'),
                                'file_type': result.get('file_extension'),
                                'processing_time': result.get('processing_time'),
                                'classification_type': result.get('classification', {}).get('document_type'),
                                'classification_confidence': result.get('classification', {}).get('confidence'),
                            }
                            summary_row.update(summary_stats)
                            summary_data.append(summary_row)
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        csv_data = summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download Summary CSV",
                            data=csv_data,
                            file_name=f"multi_doc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()